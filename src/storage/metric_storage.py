from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import asyncio
import logging
from enum import Enum
import json
import aioredis
import aioinflux
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client import InfluxDBClient, Point
import pandas as pd
import numpy as np
from dataclasses import dataclass
import pickle
import zlib
import base64

class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AggregationType(Enum):
    """Types of metric aggregation."""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    PERCENTILE = "percentile"

@dataclass
class MetricValue:
    """Container for metric values."""
    timestamp: datetime
    value: float
    labels: Dict[str, str]
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class MetricSeries:
    """Container for metric series."""
    name: str
    metric_type: MetricType
    values: List[MetricValue]
    metadata: Optional[Dict[str, Any]] = None

class MetricStorage:
    """
    Manages metric storage and retrieval.
    Implements time-series data storage with multiple backends.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize metric storage.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.influx_client = None
        self.redis_client = None
        self.metric_cache = {}
        self.aggregation_cache = {}
        
        # Configuration
        self.retention_days = config.get('retention_days', 30)
        self.cache_duration = config.get('cache_duration', 300)
        self.batch_size = config.get('batch_size', 1000)
        self.compression_threshold = config.get('compression_threshold', 1024)
        
        # Start background tasks
        asyncio.create_task(self._initialize_storage())
        asyncio.create_task(self._cleanup_loop())
        asyncio.create_task(self._cache_cleanup_loop())

    async def store_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        timestamp: Optional[datetime] = None,
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store metric value.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            timestamp: Optional timestamp
            labels: Optional labels
            metadata: Optional metadata
            
        Returns:
            Success status
        """
        try:
            # Create metric value
            metric = MetricValue(
                timestamp=timestamp or datetime.utcnow(),
                value=value,
                labels=labels or {},
                metadata=metadata
            )
            
            # Store in InfluxDB
            if self.influx_client:
                point = Point(name) \
                    .tag("type", metric_type.value) \
                    .field("value", value) \
                    .time(metric.timestamp)
                
                # Add labels as tags
                for key, value in metric.labels.items():
                    point = point.tag(key, value)
                
                await self.influx_client.write(point)
            
            # Store in Redis cache
            if self.redis_client:
                await self._store_in_cache(name, metric)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing metric: {str(e)}")
            return False

    async def store_metrics_batch(
        self,
        metrics: List[Tuple[str, float, MetricType, Optional[Dict[str, str]]]]
    ) -> bool:
        """Store multiple metrics in batch."""
        try:
            # Prepare points for InfluxDB
            points = []
            current_time = datetime.utcnow()
            
            for name, value, metric_type, labels in metrics:
                point = Point(name) \
                    .tag("type", metric_type.value) \
                    .field("value", value) \
                    .time(current_time)
                
                if labels:
                    for key, value in labels.items():
                        point = point.tag(key, value)
                
                points.append(point)
                
                # Store in Redis cache
                if self.redis_client:
                    metric = MetricValue(
                        timestamp=current_time,
                        value=value,
                        labels=labels or {}
                    )
                    await self._store_in_cache(name, metric)
            
            # Store in InfluxDB
            if self.influx_client:
                await self.influx_client.write(points)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing metrics batch: {str(e)}")
            return False

    async def get_metric(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        labels: Optional[Dict[str, str]] = None,
        aggregation: Optional[AggregationType] = None
    ) -> Optional[Union[MetricValue, MetricSeries]]:
        """Get metric values."""
        try:
            # Try cache first
            if self.redis_client:
                cached_value = await self._get_from_cache(
                    name,
                    start_time,
                    end_time,
                    labels
                )
                if cached_value is not None:
                    return cached_value
            
            # Query InfluxDB
            if self.influx_client:
                query = f'from(bucket: "{self.config["influx_bucket"]}")'
                query += f' |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})'
                query += f' |> filter(fn: (r) => r["_measurement"] == "{name}")'
                
                if labels:
                    for key, value in labels.items():
                        query += f' |> filter(fn: (r) => r["{key}"] == "{value}")'
                
                if aggregation:
                    query += self._get_aggregation_query(aggregation)
                
                result = await self.influx_client.query(query)
                
                # Process results
                values = []
                for record in result:
                    values.append(MetricValue(
                        timestamp=record.get_time(),
                        value=record.get_value(),
                        labels=record.values
                    ))
                
                if not values:
                    return None
                    
                if len(values) == 1:
                    return values[0]
                    
                return MetricSeries(
                    name=name,
                    metric_type=MetricType(record.get_field()),
                    values=values
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting metric: {str(e)}")
            return None

    async def get_metrics_aggregated(
        self,
        names: List[str],
        aggregation: AggregationType,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, List[MetricValue]]:
        """Get aggregated metrics."""
        try:
            results = {}
            
            # Check cache
            cache_key = f"agg_{','.join(names)}_{aggregation.value}_{interval}"
            cached_result = self.aggregation_cache.get(cache_key)
            if cached_result and cached_result['expires'] > datetime.utcnow():
                return cached_result['data']
            
            # Query InfluxDB
            if self.influx_client:
                for name in names:
                    query = f'from(bucket: "{self.config["influx_bucket"]}")'
                    query += f' |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})'
                    query += f' |> filter(fn: (r) => r["_measurement"] == "{name}")'
                    
                    if labels:
                        for key, value in labels.items():
                            query += f' |> filter(fn: (r) => r["{key}"] == "{value}")'
                    
                    # Add aggregation
                    query += f' |> window(every: {interval})'
                    query += self._get_aggregation_query(aggregation)
                    
                    result = await self.influx_client.query(query)
                    
                    # Process results
                    values = []
                    for record in result:
                        values.append(MetricValue(
                            timestamp=record.get_time(),
                            value=record.get_value(),
                            labels=record.values
                        ))
                    
                    if values:
                        results[name] = values
            
            # Cache results
            self.aggregation_cache[cache_key] = {
                'data': results,
                'expires': datetime.utcnow() + timedelta(seconds=self.cache_duration)
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error getting aggregated metrics: {str(e)}")
            return {}

    async def delete_metrics(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> bool:
        """Delete metrics."""
        try:
            # Delete from InfluxDB
            if self.influx_client:
                query = f'from(bucket: "{self.config["influx_bucket"]}")'
                query += f' |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})'
                query += f' |> filter(fn: (r) => r["_measurement"] == "{name}")'
                
                if labels:
                    for key, value in labels.items():
                        query += f' |> filter(fn: (r) => r["{key}"] == "{value}")'
                
                query += ' |> drop()'
                
                await self.influx_client.query(query)
            
            # Delete from Redis cache
            if self.redis_client:
                await self._delete_from_cache(name, start_time, end_time, labels)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting metrics: {str(e)}")
            return False

    async def get_metric_metadata(
        self,
        name: str
    ) -> Optional[Dict[str, Any]]:
        """Get metric metadata."""
        try:
            # Try Redis first
            if self.redis_client:
                metadata = await self.redis_client.get(f"metric_metadata:{name}")
                if metadata:
                    return json.loads(metadata)
            
            # Query InfluxDB
            if self.influx_client:
                query = f'from(bucket: "{self.config["influx_bucket"]}")'
                query += f' |> range(start: -1h)'
                query += f' |> filter(fn: (r) => r["_measurement"] == "{name}")'
                query += ' |> first()'
                
                result = await self.influx_client.query(query)
                if result:
                    record = next(result)
                    metadata = {
                        'type': record.get_field(),
                        'labels': record.values,
                        'last_update': record.get_time().isoformat()
                    }
                    
                    # Cache metadata
                    if self.redis_client:
                        await self.redis_client.set(
                            f"metric_metadata:{name}",
                            json.dumps(metadata),
                            expire=3600
                        )
                    
                    return metadata
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting metric metadata: {str(e)}")
            return None

    async def _initialize_storage(self):
        """Initialize storage backends."""
        try:
            # Initialize InfluxDB client
            if 'influxdb' in self.config:
                self.influx_client = InfluxDBClient(
                    url=self.config['influxdb']['url'],
                    token=self.config['influxdb']['token'],
                    org=self.config['influxdb']['org']
                )
            
            # Initialize Redis client
            if 'redis' in self.config:
                self.redis_client = await aioredis.create_redis_pool(
                    self.config['redis']['url'],
                    minsize=5,
                    maxsize=10
                )
            
        except Exception as e:
            self.logger.error(f"Error initializing storage: {str(e)}")

    async def _cleanup_loop(self):
        """Background task for cleaning up old metrics."""
        while True:
            try:
                # Delete old metrics from InfluxDB
                if self.influx_client:
                    cutoff_time = datetime.utcnow() - timedelta(
                        days=self.retention_days
                    )
                    
                    query = f'from(bucket: "{self.config["influx_bucket"]}")'
                    query += f' |> range(start: 0, stop: {cutoff_time.isoformat()})'
                    query += ' |> drop()'
                    
                    await self.influx_client.query(query)
                
                await asyncio.sleep(3600 * 24)  # Run daily
                
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {str(e)}")
                await asyncio.sleep(3600)

    async def _cache_cleanup_loop(self):
        """Background task for cleaning up cache."""
        while True:
            try:
                current_time = datetime.utcnow()
                
                # Clean up aggregation cache
                expired_keys = [
                    key for key, value in self.aggregation_cache.items()
                    if value['expires'] < current_time
                ]
                
                for key in expired_keys:
                    del self.aggregation_cache[key]
                
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                self.logger.error(f"Error in cache cleanup loop: {str(e)}")
                await asyncio.sleep(60)

    async def _store_in_cache(
        self,
        name: str,
        metric: MetricValue
    ):
        """Store metric in Redis cache."""
        try:
            # Serialize metric
            data = {
                'timestamp': metric.timestamp.isoformat(),
                'value': metric.value,
                'labels': metric.labels,
                'metadata': metric.metadata
            }
            
            serialized = json.dumps(data)
            
            # Compress if needed
            if len(serialized) > self.compression_threshold:
                compressed = zlib.compress(serialized.encode())
                serialized = base64.b64encode(compressed).decode()
                key = f"metric_compressed:{name}"
            else:
                key = f"metric:{name}"
            
            # Store in Redis
            await self.redis_client.set(
                key,
                serialized,
                expire=self.cache_duration
            )
            
        except Exception as e:
            self.logger.error(f"Error storing in cache: {str(e)}")

    async def _get_from_cache(
        self,
        name: str,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        labels: Optional[Dict[str, str]]
    ) -> Optional[Union[MetricValue, MetricSeries]]:
        """Get metric from Redis cache."""
        try:
            # Try compressed and uncompressed keys
            for key in [f"metric:{name}", f"metric_compressed:{name}"]:
                data = await self.redis_client.get(key)
                if data:
                    # Decompress if needed
                    if key.startswith("metric_compressed:"):
                        compressed = base64.b64decode(data)
                        data = zlib.decompress(compressed).decode()
                    
                    # Parse data
                    parsed = json.loads(data)
                    
                    # Check time range
                    timestamp = datetime.fromisoformat(parsed['timestamp'])
                    if start_time and timestamp < start_time:
                        continue
                    if end_time and timestamp > end_time:
                        continue
                    
                    # Check labels
                    if labels:
                        if not all(
                            parsed['labels'].get(k) == v
                            for k, v in labels.items()
                        ):
                            continue
                    
                    return MetricValue(
                        timestamp=timestamp,
                        value=parsed['value'],
                        labels=parsed['labels'],
                        metadata=parsed.get('metadata')
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting from cache: {str(e)}")
            return None

    def _get_aggregation_query(self, aggregation: AggregationType) -> str:
        """Get InfluxDB aggregation query part."""
        if aggregation == AggregationType.SUM:
            return ' |> sum()'
        elif aggregation == AggregationType.AVG:
            return ' |> mean()'
        elif aggregation == AggregationType.MIN:
            return ' |> min()'
        elif aggregation == AggregationType.MAX:
            return ' |> max()'
        elif aggregation == AggregationType.COUNT:
            return ' |> count()'
        elif aggregation == AggregationType.PERCENTILE:
            return ' |> quantile(q: 0.95)'
        else:
            raise ValueError(f"Unsupported aggregation type: {aggregation}")

# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'influxdb': {
            'url': 'http://localhost:8086',
            'token': 'your-token',
            'org': 'your-org',
            'bucket': 'metrics'
        },
        'redis': {
            'url': 'redis://localhost'
        },
        'retention_days': 30,
        'cache_duration': 300,
        'batch_size': 1000,
        'compression_threshold': 1024
    }
    
    # Initialize storage
    storage = MetricStorage(config)
    
    # Example operations
    async def main():
        # Store single metric
        await storage.store_metric(
            name='cpu_usage',
            value=75.5,
            metric_type=MetricType.GAUGE,
            labels={'host': 'server1', 'env': 'prod'}
        )
        
        # Store batch of metrics
        metrics = [
            ('memory_usage', 85.2, MetricType.GAUGE, {'host': 'server1'}),
            ('disk_io', 150, MetricType.COUNTER, {'host': 'server1'}),
            ('network_in', 1024, MetricType.COUNTER, {'host': 'server1'})
        ]
        await storage.store_metrics_batch(metrics)
        
        # Get metric
        metric = await storage.get_metric(
            name='cpu_usage',
            start_time=datetime.utcnow() - timedelta(hours=1),
            end_time=datetime.utcnow(),
            labels={'host': 'server1'}
        )
        
        if metric:
            print(f"CPU Usage: {metric.value}%")
        
        # Get aggregated metrics
        aggregated = await storage.get_metrics_aggregated(
            names=['cpu_usage', 'memory_usage'],
            aggregation=AggregationType.AVG,
            interval='5m',
            start_time=datetime.utcnow() - timedelta(hours=1),
            end_time=datetime.utcnow(),
            labels={'host': 'server1'}
        )
        
        for name, values in aggregated.items():
            print(f"\n{name} averages:")
            for value in values:
                print(f"  {value.timestamp}: {value.value}")
    
    # Run example
    asyncio.run(main())