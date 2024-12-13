from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from enum import Enum

class MetricType(Enum):
    """Enumeration of supported metric types."""
    GAUGE = "gauge"
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class MetricDefinition:
    """Definition of a metric with its properties."""
    name: str
    type: MetricType
    description: str
    unit: str
    labels: List[str]

class MetricsRegistry:
    """Registry of all metrics used in the system."""
    
    # System Metrics
    SYSTEM_CPU_USAGE = MetricDefinition(
        name="system_cpu_usage_percent",
        type=MetricType.GAUGE,
        description="System CPU usage percentage",
        unit="percent",
        labels=["hostname"]
    )
    
    SYSTEM_MEMORY_USAGE = MetricDefinition(
        name="system_memory_usage_percent",
        type=MetricType.GAUGE,
        description="System memory usage percentage",
        unit="percent",
        labels=["hostname"]
    )
    
    SYSTEM_DISK_USAGE = MetricDefinition(
        name="system_disk_usage_percent",
        type=MetricType.GAUGE,
        description="System disk usage percentage",
        unit="percent",
        labels=["hostname", "mount_point"]
    )
    
    # Container Metrics
    CONTAINER_CPU_USAGE = MetricDefinition(
        name="container_cpu_usage_percent",
        type=MetricType.GAUGE,
        description="Container CPU usage percentage",
        unit="percent",
        labels=["container_id", "container_name"]
    )
    
    CONTAINER_MEMORY_USAGE = MetricDefinition(
        name="container_memory_usage_percent",
        type=MetricType.GAUGE,
        description="Container memory usage percentage",
        unit="percent",
        labels=["container_id", "container_name"]
    )
    
    CONTAINER_NETWORK_RX = MetricDefinition(
        name="container_network_receive_bytes",
        type=MetricType.COUNTER,
        description="Container network bytes received",
        unit="bytes",
        labels=["container_id", "container_name", "interface"]
    )
    
    CONTAINER_NETWORK_TX = MetricDefinition(
        name="container_network_transmit_bytes",
        type=MetricType.COUNTER,
        description="Container network bytes transmitted",
        unit="bytes",
        labels=["container_id", "container_name", "interface"]
    )

class MetricProcessor:
    """Process and validate metrics data."""
    
    def __init__(self):
        self.metrics_registry = {
            metric.name: metric 
            for metric in MetricsRegistry.__dict__.values() 
            if isinstance(metric, MetricDefinition)
        }

    def validate_metric(self, name: str, value: float, labels: Dict[str, str]) -> bool:
        """
        Validate a metric value and its labels.
        
        Args:
            name: Metric name
            value: Metric value
            labels: Metric labels
            
        Returns:
            bool: True if metric is valid
        """
        if name not in self.metrics_registry:
            return False
            
        metric_def = self.metrics_registry[name]
        
        # Validate labels
        if not all(label in labels for label in metric_def.labels):
            return False
            
        # Validate value based on metric type
        if metric_def.type in [MetricType.GAUGE, MetricType.HISTOGRAM]:
            return isinstance(value, (int, float))
        elif metric_def.type == MetricType.COUNTER:
            return isinstance(value, (int, float)) and value >= 0
            
        return True

    def process_metric(self, name: str, value: float, labels: Dict[str, str]) -> Dict:
        """
        Process a metric and prepare it for storage/export.
        
        Args:
            name: Metric name
            value: Metric value
            labels: Metric labels
            
        Returns:
            Dict: Processed metric
        """
        if not self.validate_metric(name, value, labels):
            raise ValueError(f"Invalid metric: {name}")
            
        return {
            "name": name,
            "value": float(value),
            "labels": labels,
            "timestamp": datetime.utcnow().timestamp()
        }

class MetricAggregator:
    """Aggregate metrics over time windows."""
    
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.metrics_buffer: Dict[str, List] = {}

    def add_metric(self, metric: Dict):
        """Add a metric to the buffer."""
        key = f"{metric['name']}_{str(sorted(metric['labels'].items()))}"
        
        if key not in self.metrics_buffer:
            self.metrics_buffer[key] = []
            
        self.metrics_buffer[key].append({
            'value': metric['value'],
            'timestamp': metric['timestamp']
        })
        
        # Remove old metrics
        current_time = datetime.utcnow().timestamp()
        self.metrics_buffer[key] = [
            m for m in self.metrics_buffer[key]
            if current_time - m['timestamp'] <= self.window_size
        ]

    def get_aggregated_metrics(self) -> Dict[str, Dict]:
        """Get aggregated metrics for all windows."""
        result = {}
        
        for key, metrics in self.metrics_buffer.items():
            if not metrics:
                continue
                
            values = [m['value'] for m in metrics]
            result[key] = {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'avg': float(np.mean(values)),
                'std': float(np.std(values)),
                'count': len(values),
                'last_update': max(m['timestamp'] for m in metrics)
            }
            
        return result

class MetricsManager:
    """Main metrics management class."""
    
    def __init__(self, config: Dict):
        self.processor = MetricProcessor()
        self.aggregator = MetricAggregator(
            window_size=config.get('aggregation_window', 60)
        )

    def add_metric(self, name: str, value: float, labels: Dict[str, str]):
        """Add a new metric."""
        processed_metric = self.processor.process_metric(name, value, labels)
        self.aggregator.add_metric(processed_metric)

    def get_metrics(self) -> Dict[str, Dict]:
        """Get all aggregated metrics."""
        return self.aggregator.get_aggregated_metrics()

    def get_metric_definitions(self) -> Dict[str, MetricDefinition]:
        """Get all metric definitions."""
        return self.processor.metrics_registry