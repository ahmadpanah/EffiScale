# src/controllers/decision_maker.py

from typing import Dict, List, Any, Tuple
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
import logging
from dataclasses import dataclass

class ScalingDecision(Enum):
    VERTICAL_SCALE_UP = "vertical_scale_up"
    VERTICAL_SCALE_DOWN = "vertical_scale_down"
    HORIZONTAL_SCALE_OUT = "horizontal_scale_out"
    HORIZONTAL_SCALE_IN = "horizontal_scale_in"
    NO_ACTION = "no_action"

@dataclass
class ResourceThresholds:
    cpu_high: float
    cpu_low: float
    memory_high: float
    memory_low: float
    network_high: float
    network_low: float

class DecisionMaker:
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.historical_decisions: List[Dict[str, Any]] = []
        self.thresholds = ResourceThresholds(
            cpu_high=0.8,
            cpu_low=0.2,
            memory_high=0.8,
            memory_low=0.2,
            network_high=0.8,
            network_low=0.2
        )
        self.logger = self._setup_logger()
        self.scaling_cooldown = 300  # 5 minutes
        self.last_scaling_time: Dict[str, datetime] = {}

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("DecisionMaker")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    async def make_scaling_decision(self, 
                                  container_id: str, 
                                  metrics: Dict[str, float], 
                                  container_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Make a scaling decision based on current metrics and historical data
        """
        try:
            # Check cooldown period
            if not self._can_scale(container_id):
                return self._create_decision_response(
                    ScalingDecision.NO_ACTION,
                    "Cooling down from previous scaling action"
                )

            # Update thresholds based on historical data
            self._update_adaptive_thresholds(container_history)

            # Analyze current metrics and predict trend
            current_state = self._analyze_current_state(metrics)
            trend = self._predict_trend(container_history)

            # Make decision based on current state and trend
            decision = self._decide_scaling_action(current_state, trend)

            # Record decision
            self._record_decision(container_id, metrics, decision)

            return self._create_decision_response(decision, "Decision based on metrics and trends")

        except Exception as e:
            self.logger.error(f"Error making scaling decision: {str(e)}")
            return self._create_decision_response(
                ScalingDecision.NO_ACTION,
                f"Error in decision making: {str(e)}"
            )

    def _can_scale(self, container_id: str) -> bool:
        """Check if container can be scaled based on cooldown period"""
        if container_id not in self.last_scaling_time:
            return True

        time_since_last_scaling = (
            datetime.now() - self.last_scaling_time[container_id]
        ).total_seconds()
        return time_since_last_scaling >= self.scaling_cooldown

    def _analyze_current_state(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """Analyze current resource utilization state"""
        state = {}
        
        # CPU analysis
        if metrics.get('cpu_usage', 0) > self.thresholds.cpu_high:
            state['cpu'] = 'high'
        elif metrics.get('cpu_usage', 0) < self.thresholds.cpu_low:
            state['cpu'] = 'low'
        else:
            state['cpu'] = 'normal'

        # Memory analysis
        if metrics.get('memory_usage', 0) > self.thresholds.memory_high:
            state['memory'] = 'high'
        elif metrics.get('memory_usage', 0) < self.thresholds.memory_low:
            state['memory'] = 'low'
        else:
            state['memory'] = 'normal'

        # Network analysis
        if metrics.get('network_usage', 0) > self.thresholds.network_high:
            state['network'] = 'high'
        elif metrics.get('network_usage', 0) < self.thresholds.network_low:
            state['network'] = 'low'
        else:
            state['network'] = 'normal'

        return state

    def _predict_trend(self, history: List[Dict[str, Any]]) -> Dict[str, str]:
        """Predict resource utilization trends based on historical data"""
        if not history:
            return {'cpu': 'stable', 'memory': 'stable', 'network': 'stable'}

        recent_history = history[-10:]  # Consider last 10 data points
        
        trends = {}
        for metric in ['cpu_usage', 'memory_usage', 'network_usage']:
            values = [entry.get('metrics', {}).get(metric, 0) for entry in recent_history]
            if len(values) < 2:
                trends[metric.split('_')[0]] = 'stable'
                continue

            # Calculate trend using linear regression
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            
            if slope > 0.1:
                trends[metric.split('_')[0]] = 'increasing'
            elif slope < -0.1:
                trends[metric.split('_')[0]] = 'decreasing'
            else:
                trends[metric.split('_')[0]] = 'stable'

        return trends

    def _decide_scaling_action(self, 
                             current_state: Dict[str, str], 
                             trend: Dict[str, str]) -> ScalingDecision:
        """Determine appropriate scaling action based on state and trend"""
        
        # Check for immediate horizontal scaling needs
        if (current_state['cpu'] == 'high' and trend['cpu'] == 'increasing') or \
           (current_state['memory'] == 'high' and trend['memory'] == 'increasing'):
            return ScalingDecision.HORIZONTAL_SCALE_OUT

        if (current_state['cpu'] == 'low' and trend['cpu'] == 'decreasing' and
            current_state['memory'] == 'low' and trend['memory'] == 'decreasing'):
            return ScalingDecision.HORIZONTAL_SCALE_IN

        # Check for vertical scaling needs
        if (current_state['cpu'] == 'high' or current_state['memory'] == 'high') and \
           (trend['cpu'] == 'stable' or trend['memory'] == 'stable'):
            return ScalingDecision.VERTICAL_SCALE_UP

        if (current_state['cpu'] == 'low' and current_state['memory'] == 'low') and \
           (trend['cpu'] == 'stable' and trend['memory'] == 'stable'):
            return ScalingDecision.VERTICAL_SCALE_DOWN

        return ScalingDecision.NO_ACTION

    def _update_adaptive_thresholds(self, history: List[Dict[str, Any]]) -> None:
        """Update thresholds based on historical performance"""
        if not history:
            return

        recent_history = history[-50:]  # Consider last 50 data points
        
        # Calculate new thresholds based on historical patterns
        cpu_values = [h.get('metrics', {}).get('cpu_usage', 0) for h in recent_history]
        memory_values = [h.get('metrics', {}).get('memory_usage', 0) for h in recent_history]
        
        if cpu_values:
            cpu_std = np.std(cpu_values)
            self.thresholds.cpu_high = min(0.9, 0.8 + cpu_std)
            self.thresholds.cpu_low = max(0.1, 0.2 - cpu_std)

        if memory_values:
            memory_std = np.std(memory_values)
            self.thresholds.memory_high = min(0.9, 0.8 + memory_std)
            self.thresholds.memory_low = max(0.1, 0.2 - memory_std)

    def _record_decision(self, 
                        container_id: str, 
                        metrics: Dict[str, float], 
                        decision: ScalingDecision) -> None:
        """Record scaling decision for historical analysis"""
        self.historical_decisions.append({
            'container_id': container_id,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'decision': decision.value
        })

        if decision != ScalingDecision.NO_ACTION:
            self.last_scaling_time[container_id] = datetime.now()

    def _create_decision_response(self, 
                                decision: ScalingDecision, 
                                reason: str) -> Dict[str, Any]:
        """Create formatted decision response"""
        return {
            'decision': decision.value,
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'thresholds': {
                'cpu_high': self.thresholds.cpu_high,
                'cpu_low': self.thresholds.cpu_low,
                'memory_high': self.thresholds.memory_high,
                'memory_low': self.thresholds.memory_low,
                'network_high': self.thresholds.network_high,
                'network_low': self.thresholds.network_low
            }
        }

    def get_decision_history(self, 
                           container_id: str, 
                           time_window: int = 3600) -> List[Dict[str, Any]]:
        """Get decision history for a specific container"""
        cutoff_time = datetime.now() - timedelta(seconds=time_window)
        return [
            decision for decision in self.historical_decisions
            if (decision['container_id'] == container_id and
                datetime.fromisoformat(decision['timestamp']) > cutoff_time)
        ]

    def analyze_decision_effectiveness(self, 
                                    container_id: str, 
                                    time_window: int = 3600) -> Dict[str, Any]:
        """Analyze the effectiveness of past scaling decisions"""
        decisions = self.get_decision_history(container_id, time_window)
        
        if not decisions:
            return {'status': 'No decisions in specified time window'}

        analysis = {
            'total_decisions': len(decisions),
            'decision_breakdown': {},
            'average_metrics_before_scaling': {},
            'average_metrics_after_scaling': {}
        }

        # Calculate decision breakdown
        for decision in decisions:
            decision_type = decision['decision']
            analysis['decision_breakdown'][decision_type] = \
                analysis['decision_breakdown'].get(decision_type, 0) + 1

        return analysis

# Example usage
if __name__ == "__main__":
    # Create decision maker instance
    decision_maker = DecisionMaker()

    # Example metrics
    metrics = {
        'cpu_usage': 0.85,
        'memory_usage': 0.75,
        'network_usage': 0.60
    }

    # Example container history
    container_history = [
        {
            'metrics': {
                'cpu_usage': 0.82,
                'memory_usage': 0.73,
                'network_usage': 0.58
            },
            'timestamp': '2024-01-01T10:00:00'
        },
        # Add more historical data points...
    ]

    # Make scaling decision
    import asyncio
    decision = asyncio.run(
        decision_maker.make_scaling_decision(
            'container_123',
            metrics,
            container_history
        )
    )
    print(f"Scaling decision: {decision}")