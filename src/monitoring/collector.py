import asyncio
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import docker
import psutil
from prometheus_client import start_http_server

from .metrics import MetricsManager
from .prometheus import PrometheusManager
from ..core.exceptions import MetricCollectionError
from ..core.utils import TimeUtils, ResourceUtils

class MetricCollector:
    """
    Main metric collection service for EffiScale.
    Handles collection, processing, and storage of system and container metrics.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the metric collector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Collection settings
        self.collection_interval = config.get('monitoring', {}).get('interval', 5)
        self.batch_size = config.get('monitoring', {}).get('batch_size', 100)
        
        # Initialize components
        self.metrics_manager = MetricsManager(config)
        self.prometheus_manager = PrometheusManager(config)
        
        # Initialize Docker client
        self.docker_client = docker.from_env()
        
        # Collection state
        self._is_collecting = False
        self._monitored_containers: Set[str] = set()
        self._last_collection_time: Dict[str, datetime] = {}
        
        # Thread pool for parallel collection
        self.executor = ThreadPoolExecutor(
            max_workers=config.get('monitoring', {}).get('max_workers', 5)
        )
        
        # Initialize metrics cache
        self._metrics_cache: Dict[str, Dict] = {}
        
        # Setup metric exporters
        self._setup_exporters()

    def _setup_exporters(self):
        """Setup metric exporters."""
        # Start Prometheus HTTP server
        prometheus_port = self.config.get('prometheus', {}).get('port', 8000)
        start_http_server(prometheus_port)

    async def start(self):
        """Start the metric collection process."""
        self.logger.info("Starting metric collection service")
        self._is_collecting = True
        
        try:
            while self._is_collecting:
                await self._collection_cycle()
                await asyncio.sleep(self.collection_interval)
                
        except Exception as e:
            self.logger.error(f"Error in metric collection cycle: {str(e)}")
            self._is_collecting = False
            raise MetricCollectionError("Metric collection failed", str(e))

    async def stop(self):
        """Stop the metric collection process."""
        self.logger.info("Stopping metric collection service")
        self._is_collecting = False
        self.executor.shutdown(wait=True)

    async def _collection_cycle(self):
        """Run a complete metric collection cycle."""
        try:
            # Update container list
            await self._update_container_list()
            
            # Collect system metrics
            await self._collect_system_metrics()
            
            # Collect container metrics
            collection_tasks = [
                self._collect_container_metrics(container_id)
                for container_id in self._monitored_containers
            ]
            
            # Run container metric collection in parallel
            await asyncio.gather(*collection_tasks)
            
            # Export metrics
            await self._export_metrics()
            
        except Exception as e:
            self.logger.error(f"Error in collection cycle: {str(e)}")
            raise MetricCollectionError("Collection cycle failed", str(e))

    async def _update_container_list(self):
        """Update the list of monitored containers."""
        try:
            containers = self.docker_client.containers.list(
                filters={'status': 'running'}
            )
            
            current_containers = {
                container.id for container in containers
            }
            
            # Add new containers
            new_containers = current_containers - self._monitored_containers
            for container_id in new_containers:
                self.logger.info(f"Adding new container to monitoring: {container_id}")
                self._monitored_containers.add(container_id)
            
            # Remove stopped containers
            stopped_containers = self._monitored_containers - current_containers
            for container_id in stopped_containers:
                self.logger.info(f"Removing stopped container from monitoring: {container_id}")
                self._monitored_containers.remove(container_id)
                self._last_collection_time.pop(container_id, None)
                
        except Exception as e:
            self.logger.error(f"Error updating container list: {str(e)}")
            raise MetricCollectionError("Container list update failed", str(e))

    async def _collect_system_metrics(self):
        """Collect system-wide metrics."""
        try:
            # Collect CPU metrics
            cpu_metrics = {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'cpu_count': psutil.cpu_count(),
                'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0
            }
            
            # Collect memory metrics
            memory = psutil.virtual_memory()
            memory_metrics = {
                'memory_total': memory.total,
                'memory_available': memory.available,
                'memory_used': memory.used,
                'memory_percent': memory.percent
            }
            
            # Collect disk metrics
            disk = psutil.disk_usage('/')
            disk_metrics = {
                'disk_total': disk.total,
                'disk_used': disk.used,
                'disk_free': disk.free,
                'disk_percent': disk.percent
            }
            
            # Collect network metrics
            network = psutil.net_io_counters()
            network_metrics = {
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'network_packets_sent': network.packets_sent,
                'network_packets_recv': network.packets_recv
            }
            
            # Process and store metrics
            timestamp = TimeUtils.get_utc_now()
            
            for name, value in {**cpu_metrics, **memory_metrics, **disk_metrics, **network_metrics}.items():
                self.metrics_manager.add_metric(
                    name=f"system_{name}",
                    value=float(value),
                    labels={'hostname': self.config.get('hostname', 'unknown')}
                )
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {str(e)}")
            raise MetricCollectionError("System metrics collection failed", str(e))

    async def _collect_container_metrics(self, container_id: str):
        """
        Collect metrics for a specific container.
        
        Args:
            container_id: Container ID to collect metrics from
        """
        try:
            # Get container
            container = self.docker_client.containers.get(container_id)
            
            # Get container stats
            stats = container.stats(stream=False)
            
            # Process CPU metrics
            cpu_metrics = {
                'cpu_usage_percent': ResourceUtils.calculate_cpu_percentage(stats),
                'cpu_system_usage': stats['cpu_stats']['system_cpu_usage'],
                'cpu_online_cpus': stats['cpu_stats'].get('online_cpus', 0)
            }
            
            # Process memory metrics
            memory_metrics = {
                'memory_usage': stats['memory_stats'].get('usage', 0),
                'memory_limit': stats['memory_stats'].get('limit', 0),
                'memory_percent': ResourceUtils.calculate_memory_percentage(stats['memory_stats'])
            }
            
            # Process network metrics
            network_metrics = {}
            if 'networks' in stats:
                for interface, data in stats['networks'].items():
                    network_metrics.update({
                        f'network_{interface}_rx_bytes': data['rx_bytes'],
                        f'network_{interface}_tx_bytes': data['tx_bytes'],
                        f'network_{interface}_rx_packets': data['rx_packets'],
                        f'network_{interface}_tx_packets': data['tx_packets']
                    })
            
            # Process block I/O metrics
            blkio_metrics = {}
            if 'blkio_stats' in stats:
                for stat in stats['blkio_stats'].get('io_service_bytes_recursive', []):
                    op = stat.get('op', '').lower()
                    if op in ['read', 'write']:
                        blkio_metrics[f'blkio_{op}_bytes'] = stat.get('value', 0)
            
            # Prepare labels
            labels = {
                'container_id': container_id,
                'container_name': container.name,
                'image': container.image.tags[0] if container.image.tags else 'unknown'
            }
            
            # Store metrics
            timestamp = TimeUtils.get_utc_now()
            
            for name, value in {**cpu_metrics, **memory_metrics, **network_metrics, **blkio_metrics}.items():
                self.metrics_manager.add_metric(
                    name=f"container_{name}",
                    value=float(value),
                    labels=labels
                )
            
            # Update last collection time
            self._last_collection_time[container_id] = timestamp
            
        except Exception as e:
            self.logger.error(f"Error collecting container metrics for {container_id}: {str(e)}")
            raise MetricCollectionError(f"Container metrics collection failed for {container_id}", str(e))

    async def _export_metrics(self):
        """Export collected metrics to configured destinations."""
        try:
            # Get all current metrics
            metrics = self.metrics_manager.get_metrics()
            
            # Export to Prometheus
            for metric_key, metric_data in metrics.items():
                name = metric_data['name']
                value = metric_data['value']
                labels = metric_data['labels']
                
                self.prometheus_manager.update_metric(
                    name=name,
                    value=value,
                    labels=labels
                )
            
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {str(e)}")
            raise MetricCollectionError("Metrics export failed", str(e))

    async def get_metrics(
        self,
        metric_type: Optional[str] = None,
        container_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get collected metrics.
        
        Args:
            metric_type: Optional filter by metric type
            container_id: Optional filter by container ID
            start_time: Optional start time for historical data
            end_time: Optional end time for historical data
            
        Returns:
            List of metrics matching the criteria
        """
        try:
            if start_time and end_time:
                # Get historical data from Prometheus
                return await self.prometheus_manager.query_metric_history(
                    metric_type=metric_type,
                    start_time=start_time,
                    end_time=end_time,
                    labels={'container_id': container_id} if container_id else None
                )
            else:
                # Get current metrics from memory
                metrics = self.metrics_manager.get_metrics()
                
                # Apply filters
                filtered_metrics = []
                for metric in metrics:
                    if metric_type and not metric['name'].startswith(metric_type):
                        continue
                    if container_id and metric['labels'].get('container_id') != container_id:
                        continue
                    filtered_metrics.append(metric)
                
                return filtered_metrics
                
        except Exception as e:
            self.logger.error(f"Error retrieving metrics: {str(e)}")
            return []

# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'monitoring': {
            'interval': 5,
            'batch_size': 100,
            'max_workers': 5
        },
        'prometheus': {
            'port': 8000,
            'url': 'http://localhost:9090'
        },
        'hostname': 'example-host'
    }
    
    # Initialize and start collector
    async def main():
        collector = MetricCollector(config)
        try:
            await collector.start()
        except KeyboardInterrupt:
            await collector.stop()
    
    # Run the collector
    asyncio.run(main())