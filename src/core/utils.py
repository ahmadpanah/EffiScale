import os
import json
import yaml
import time
import uuid
import hashlib
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timezone
from functools import wraps
import docker
import psutil
import numpy as np
from .exceptions import ConfigurationError, ValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TimeUtils:
    """Utility class for time-related operations."""
    
    @staticmethod
    def get_utc_now() -> datetime:
        """Get current UTC timestamp."""
        return datetime.now(timezone.utc)
    
    @staticmethod
    def format_timestamp(dt: datetime) -> str:
        """Format datetime to ISO 8601 string."""
        return dt.isoformat()
    
    @staticmethod
    def parse_timestamp(timestamp_str: str) -> datetime:
        """Parse ISO 8601 timestamp string to datetime."""
        return datetime.fromisoformat(timestamp_str)
    
    @staticmethod
    def calculate_time_window(start: datetime, end: datetime) -> float:
        """Calculate time window in seconds."""
        return (end - start).total_seconds()

class ResourceUtils:
    """Utility class for resource-related operations."""
    
    @staticmethod
    def calculate_cpu_percentage(cpu_stats: Dict) -> float:
        """Calculate CPU usage percentage from Docker stats."""
        cpu_delta = cpu_stats['cpu_stats']['cpu_usage']['total_usage'] - \
                   cpu_stats['precpu_stats']['cpu_usage']['total_usage']
        system_delta = cpu_stats['cpu_stats']['system_cpu_usage'] - \
                      cpu_stats['precpu_stats']['system_cpu_usage']
        
        if system_delta > 0 and cpu_delta > 0:
            return (cpu_delta / system_delta) * 100.0
        return 0.0
    
    @staticmethod
    def calculate_memory_percentage(memory_stats: Dict) -> float:
        """Calculate memory usage percentage from Docker stats."""
        if 'usage' in memory_stats and 'limit' in memory_stats:
            return (memory_stats['usage'] / memory_stats['limit']) * 100.0
        return 0.0
    
    @staticmethod
    def get_system_resources() -> Dict[str, float]:
        """Get current system resource usage."""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent
        }

class ValidationUtils:
    """Utility class for data validation operations."""
    
    @staticmethod
    def validate_metrics(metrics: Dict) -> bool:
        """Validate metric data structure."""
        required_fields = {'timestamp', 'container_id', 'cpu_usage', 'memory_usage'}
        return all(field in metrics for field in required_fields)
    
    @staticmethod
    def validate_scaling_decision(decision: Dict) -> bool:
        """Validate scaling decision structure."""
        required_fields = {'action', 'resource_type', 'target_value'}
        return all(field in decision for field in required_fields)
    
    @staticmethod
    def validate_config(config: Dict) -> bool:
        """Validate configuration structure."""
        required_fields = {
            'monitoring': {'interval', 'metrics'},
            'scaling': {'thresholds', 'cooldown_period'},
            'controllers': {'count', 'consensus_threshold'}
        }
        
        try:
            for section, fields in required_fields.items():
                if section not in config:
                    raise ValidationError(f"Missing section: {section}")
                for field in fields:
                    if field not in config[section]:
                        raise ValidationError(f"Missing field: {field} in section {section}")
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            return False

class DockerUtils:
    """Utility class for Docker-related operations."""
    
    def __init__(self):
        self.client = docker.from_client()
    
    def get_container_stats(self, container_id: str) -> Dict:
        """Get container statistics."""
        try:
            container = self.client.containers.get(container_id)
            return container.stats(stream=False)
        except Exception as e:
            logger.error(f"Failed to get container stats: {str(e)}")
            return {}
    
    def scale_container(self, container_id: str, cpu_quota: int = None, memory: int = None) -> bool:
        """Scale container resources."""
        try:
            container = self.client.containers.get(container_id)
            update_config = {}
            
            if cpu_quota is not None:
                update_config['cpu_quota'] = cpu_quota
            if memory is not None:
                update_config['mem_limit'] = memory
                
            container.update(**update_config)
            return True
        except Exception as e:
            logger.error(f"Failed to scale container: {str(e)}")
            return False

class MetricsUtils:
    """Utility class for metrics-related operations."""
    
    @staticmethod
    def calculate_moving_average(data: List[float], window: int = 5) -> List[float]:
        """Calculate moving average of metrics."""
        return list(np.convolve(data, np.ones(window)/window, mode='valid'))
    
    @staticmethod
    def detect_anomalies(data: List[float], threshold: float = 2.0) -> List[int]:
        """Detect anomalies in metric data using Z-score."""
        mean = np.mean(data)
        std = np.std(data)
        z_scores = [(x - mean) / std for x in data]
        return [i for i, z in enumerate(z_scores) if abs(z) > threshold]
    
    @staticmethod
    def normalize_metrics(data: List[float]) -> List[float]:
        """Normalize metric values to range [0, 1]."""
        min_val = min(data)
        max_val = max(data)
        if max_val == min_val:
            return [0.5] * len(data)
        return [(x - min_val) / (max_val - min_val) for x in data]

def retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
    """Decorator for retrying operations on failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        logger.error(f"Operation failed after {max_attempts} attempts: {str(e)}")
                        raise
                    logger.warning(f"Attempt {attempts} failed, retrying after {delay} seconds")
                    time.sleep(delay)
        return wrapper
    return decorator

def generate_unique_id() -> str:
    """Generate unique identifier."""
    return str(uuid.uuid4())

def hash_data(data: Any) -> str:
    """Generate hash of data."""
    return hashlib.sha256(str(data).encode()).hexdigest()

def load_yaml_config(file_path: str) -> Dict:
    """Load YAML configuration file."""
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration file: {str(e)}")
        raise ConfigurationError(f"Failed to load configuration from {file_path}")

def save_to_json(data: Any, file_path: str) -> bool:
    """Save data to JSON file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON file: {str(e)}")
        return False

@retry_on_failure(max_attempts=3)
def ensure_directory(directory: str) -> bool:
    """Ensure directory exists, create if it doesn't."""
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory: {str(e)}")
        return False

# Example usage
if __name__ == "__main__":
    # Time utilities
    current_time = TimeUtils.get_utc_now()
    formatted_time = TimeUtils.format_timestamp(current_time)
    print(f"Current UTC time: {formatted_time}")
    
    # Resource utilities
    system_resources = ResourceUtils.get_system_resources()
    print(f"System resources: {system_resources}")
    
    # Metrics utilities
    data = [1.0, 2.0, 3.0, 10.0, 2.0, 3.0]
    moving_avg = MetricsUtils.calculate_moving_average(data)
    anomalies = MetricsUtils.detect_anomalies(data)
    print(f"Moving average: {moving_avg}")
    print(f"Anomalies detected at indices: {anomalies}")