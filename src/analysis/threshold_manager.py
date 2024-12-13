from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass
import json
import logging
from enum import Enum
import asyncio
from abc import ABC, abstractmethod

class ThresholdType(Enum):
    """Types of thresholds supported by the system."""
    STATIC = "static"
    DYNAMIC = "dynamic"
    ADAPTIVE = "adaptive"
    TIME_BASED = "time_based"
    PERCENTILE = "percentile"

class ComparisonOperator(Enum):
    """Comparison operators for threshold evaluation."""
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="

@dataclass
class ThresholdConfig:
    """Configuration for a threshold."""
    metric_name: str
    threshold_type: ThresholdType
    operator: ComparisonOperator
    value: float
    adaptation_rate: Optional[float] = 0.1
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    time_windows: Optional[Dict[str, float]] = None
    percentile: Optional[float] = None

@dataclass
class ThresholdState:
    """Current state of a threshold."""
    current_value: float
    last_update: datetime
    violation_count: int
    total_evaluations: int
    adaptation_history: List[Tuple[datetime, float]]

class ThresholdEvaluator(ABC):
    """Abstract base class for threshold evaluators."""
    
    @abstractmethod
    def evaluate(self, value: float, threshold: float) -> bool:
        """Evaluate if value violates threshold."""
        pass

class ThresholdManager:
    """
    Manages dynamic thresholds for metrics monitoring.
    Supports multiple threshold types and adaptive behavior.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the threshold manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize threshold configurations
        self.thresholds: Dict[str, ThresholdConfig] = {}
        self.threshold_states: Dict[str, ThresholdState] = {}
        
        # Load initial thresholds
        self._load_thresholds()
        
        # Initialize evaluators
        self.evaluators = {
            ComparisonOperator.GREATER_THAN: lambda x, t: x > t,
            ComparisonOperator.LESS_THAN: lambda x, t: x < t,
            ComparisonOperator.GREATER_EQUAL: lambda x, t: x >= t,
            ComparisonOperator.LESS_EQUAL: lambda x, t: x <= t,
            ComparisonOperator.EQUAL: lambda x, t: abs(x - t) < 1e-6,
            ComparisonOperator.NOT_EQUAL: lambda x, t: abs(x - t) >= 1e-6
        }
        
        # Metrics history for adaptation
        self.metrics_history: Dict[str, List[Tuple[datetime, float]]] = {}
        
        # Configuration for adaptation
        self.adaptation_window = config.get('adaptation', {}).get('window_size', 3600)  # 1 hour
        self.adaptation_interval = config.get('adaptation', {}).get('interval', 300)    # 5 minutes
        
        # Start adaptation task
        if config.get('adaptation', {}).get('enabled', True):
            asyncio.create_task(self._adaptation_loop())

    def _load_thresholds(self):
        """Load threshold configurations from config."""
        threshold_configs = self.config.get('thresholds', {})
        
        for metric_name, config in threshold_configs.items():
            try:
                threshold_config = ThresholdConfig(
                    metric_name=metric_name,
                    threshold_type=ThresholdType(config['type']),
                    operator=ComparisonOperator(config['operator']),
                    value=float(config['value']),
                    adaptation_rate=config.get('adaptation_rate', 0.1),
                    min_value=config.get('min_value'),
                    max_value=config.get('max_value'),
                    time_windows=config.get('time_windows'),
                    percentile=config.get('percentile')
                )
                
                self.thresholds[metric_name] = threshold_config
                self.threshold_states[metric_name] = ThresholdState(
                    current_value=threshold_config.value,
                    last_update=datetime.utcnow(),
                    violation_count=0,
                    total_evaluations=0,
                    adaptation_history=[]
                )
                
            except (ValueError, KeyError) as e:
                self.logger.error(f"Error loading threshold config for {metric_name}: {str(e)}")

    async def evaluate_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None
    ) -> Tuple[bool, Dict]:
        """
        Evaluate a metric value against its threshold.
        
        Args:
            metric_name: Name of the metric
            value: Metric value to evaluate
            timestamp: Optional timestamp of the metric
            
        Returns:
            Tuple of (violation_status, evaluation_details)
        """
        if metric_name not in self.thresholds:
            raise ValueError(f"No threshold configured for metric: {metric_name}")
            
        timestamp = timestamp or datetime.utcnow()
        threshold_config = self.thresholds[metric_name]
        threshold_state = self.threshold_states[metric_name]
        
        # Update metrics history
        self._update_metrics_history(metric_name, value, timestamp)
        
        # Get current threshold value
        current_threshold = self._get_current_threshold(metric_name, timestamp)
        
        # Evaluate threshold
        violation = self.evaluators[threshold_config.operator](value, current_threshold)
        
        # Update state
        threshold_state.total_evaluations += 1
        if violation:
            threshold_state.violation_count += 1
            
        # Prepare evaluation details
        details = {
            'threshold_value': current_threshold,
            'threshold_type': threshold_config.threshold_type.value,
            'operator': threshold_config.operator.value,
            'violation_rate': threshold_state.violation_count / threshold_state.total_evaluations,
            'last_update': threshold_state.last_update.isoformat(),
            'adaptation_history': [
                {'timestamp': ts.isoformat(), 'value': val}
                for ts, val in threshold_state.adaptation_history[-5:]  # Last 5 adaptations
            ]
        }
        
        return violation, details

    def _update_metrics_history(
        self,
        metric_name: str,
        value: float,
        timestamp: datetime
    ):
        """Update metrics history for adaptation."""
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = []
            
        self.metrics_history[metric_name].append((timestamp, value))
        
        # Remove old entries
        cutoff_time = timestamp - timedelta(seconds=self.adaptation_window)
        self.metrics_history[metric_name] = [
            (ts, val) for ts, val in self.metrics_history[metric_name]
            if ts > cutoff_time
        ]

    def _get_current_threshold(
        self,
        metric_name: str,
        timestamp: datetime
    ) -> float:
        """Get current threshold value based on type."""
        threshold_config = self.thresholds[metric_name]
        threshold_state = self.threshold_states[metric_name]
        
        if threshold_config.threshold_type == ThresholdType.STATIC:
            return threshold_config.value
            
        elif threshold_config.threshold_type == ThresholdType.TIME_BASED:
            return self._get_time_based_threshold(threshold_config, timestamp)
            
        elif threshold_config.threshold_type == ThresholdType.PERCENTILE:
            return self._get_percentile_threshold(metric_name, threshold_config)
            
        return threshold_state.current_value

    def _get_time_based_threshold(
        self,
        config: ThresholdConfig,
        timestamp: datetime
    ) -> float:
        """Get threshold value based on time windows."""
        if not config.time_windows:
            return config.value
            
        hour = timestamp.hour
        for time_range, value in config.time_windows.items():
            start_hour, end_hour = map(int, time_range.split('-'))
            if start_hour <= hour < end_hour:
                return value
                
        return config.value

    def _get_percentile_threshold(
        self,
        metric_name: str,
        config: ThresholdConfig
    ) -> float:
        """Get threshold value based on percentile."""
        if not config.percentile or metric_name not in self.metrics_history:
            return config.value
            
        values = [val for _, val in self.metrics_history[metric_name]]
        if not values:
            return config.value
            
        return float(np.percentile(values, config.percentile))

    async def _adaptation_loop(self):
        """Background task for threshold adaptation."""
        while True:
            try:
                await self._adapt_thresholds()
                await asyncio.sleep(self.adaptation_interval)
            except Exception as e:
                self.logger.error(f"Error in threshold adaptation: {str(e)}")

    async def _adapt_thresholds(self):
        """Adapt thresholds based on recent metrics."""
        for metric_name, threshold_config in self.thresholds.items():
            if threshold_config.threshold_type != ThresholdType.ADAPTIVE:
                continue
                
            try:
                threshold_state = self.threshold_states[metric_name]
                
                if metric_name not in self.metrics_history:
                    continue
                    
                # Calculate new threshold
                recent_values = [
                    val for _, val in self.metrics_history[metric_name]
                ]
                
                if not recent_values:
                    continue
                    
                current_mean = np.mean(recent_values)
                current_std = np.std(recent_values)
                
                # Adapt threshold
                new_threshold = current_mean + (2 * current_std)  # Example adaptation strategy
                
                # Apply adaptation rate
                adapted_threshold = (
                    threshold_state.current_value * (1 - threshold_config.adaptation_rate) +
                    new_threshold * threshold_config.adaptation_rate
                )
                
                # Apply bounds
                if threshold_config.min_value is not None:
                    adapted_threshold = max(adapted_threshold, threshold_config.min_value)
                if threshold_config.max_value is not None:
                    adapted_threshold = min(adapted_threshold, threshold_config.max_value)
                
                # Update state
                threshold_state.current_value = adapted_threshold
                threshold_state.last_update = datetime.utcnow()
                threshold_state.adaptation_history.append(
                    (datetime.utcnow(), adapted_threshold)
                )
                
                # Keep only recent adaptation history
                threshold_state.adaptation_history = threshold_state.adaptation_history[-100:]
                
            except Exception as e:
                self.logger.error(f"Error adapting threshold for {metric_name}: {str(e)}")

    def add_threshold(self, config: Dict):
        """Add a new threshold configuration."""
        try:
            threshold_config = ThresholdConfig(
                metric_name=config['metric_name'],
                threshold_type=ThresholdType(config['type']),
                operator=ComparisonOperator(config['operator']),
                value=float(config['value']),
                adaptation_rate=config.get('adaptation_rate', 0.1),
                min_value=config.get('min_value'),
                max_value=config.get('max_value'),
                time_windows=config.get('time_windows'),
                percentile=config.get('percentile')
            )
            
            self.thresholds[config['metric_name']] = threshold_config
            self.threshold_states[config['metric_name']] = ThresholdState(
                current_value=threshold_config.value,
                last_update=datetime.utcnow(),
                violation_count=0,
                total_evaluations=0,
                adaptation_history=[]
            )
            
        except (ValueError, KeyError) as e:
            raise ValueError(f"Invalid threshold configuration: {str(e)}")

    def remove_threshold(self, metric_name: str):
        """Remove a threshold configuration."""
        self.thresholds.pop(metric_name, None)
        self.threshold_states.pop(metric_name, None)
        self.metrics_history.pop(metric_name, None)

    def get_threshold_status(self, metric_name: str) -> Dict:
        """Get current status of a threshold."""
        if metric_name not in self.thresholds:
            raise ValueError(f"No threshold configured for metric: {metric_name}")
            
        threshold_config = self.thresholds[metric_name]
        threshold_state = self.threshold_states[metric_name]
        
        return {
            'metric_name': metric_name,
            'threshold_type': threshold_config.threshold_type.value,
            'current_value': threshold_state.current_value,
            'original_value': threshold_config.value,
            'operator': threshold_config.operator.value,
            'violation_rate': (
                threshold_state.violation_count / threshold_state.total_evaluations
                if threshold_state.total_evaluations > 0 else 0
            ),
            'last_update': threshold_state.last_update.isoformat(),
            'adaptation_history': [
                {'timestamp': ts.isoformat(), 'value': val}
                for ts, val in threshold_state.adaptation_history[-5:]
            ]
        }

# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'thresholds': {
            'cpu_usage': {
                'type': 'adaptive',
                'operator': '>',
                'value': 80.0,
                'adaptation_rate': 0.1,
                'min_value': 60.0,
                'max_value': 95.0
            },
            'memory_usage': {
                'type': 'time_based',
                'operator': '>',
                'value': 75.0,
                'time_windows': {
                    '0-6': 85.0,    # Higher threshold during night
                    '6-18': 75.0,   # Normal threshold during day
                    '18-24': 80.0   # Medium threshold during evening
                }
            }
        },
        'adaptation': {
            'enabled': True,
            'window_size': 3600,
            'interval': 300
        }
    }
    
    # Initialize manager
    manager = ThresholdManager(config)
    
    # Example evaluation
    async def main():
        # Evaluate some metrics
        violation, details = await manager.evaluate_metric('cpu_usage', 85.0)
        print(f"CPU Usage Violation: {violation}")
        print(f"Details: {json.dumps(details, indent=2)}")
        
        # Get threshold status
        status = manager.get_threshold_status('cpu_usage')
        print(f"Threshold Status: {json.dumps(status, indent=2)}")
    
    # Run example
    asyncio.run(main())