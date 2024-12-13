from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import logging
import asyncio
from enum import Enum

class OptimizationStrategy(Enum):
    """Supported optimization strategies."""
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    COST_FOCUSED = "cost_focused"
    PERFORMANCE_FOCUSED = "performance_focused"

@dataclass
class ResourceLimits:
    """Resource limits configuration."""
    min_cpu: float
    max_cpu: float
    min_memory: float
    max_memory: float
    step_cpu: float
    step_memory: float

@dataclass
class OptimizationResult:
    """Container for optimization results."""
    cpu_allocation: float
    memory_allocation: float
    confidence: float
    predicted_performance: float
    cost_savings: float
    risk_level: float
    recommendations: List[str]

class ResourceOptimizer:
    """
    Optimizes resource allocation based on workload patterns,
    performance requirements, and cost constraints.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the resource optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize settings
        self.strategy = OptimizationStrategy(
            config.get('strategy', 'balanced')
        )
        
        # Resource limits
        self.resource_limits = ResourceLimits(
            min_cpu=config.get('resources', {}).get('min_cpu', 0.1),
            max_cpu=config.get('resources', {}).get('max_cpu', 4.0),
            min_memory=config.get('resources', {}).get('min_memory', 128),
            max_memory=config.get('resources', {}).get('max_memory', 8192),
            step_cpu=config.get('resources', {}).get('step_cpu', 0.1),
            step_memory=config.get('resources', {}).get('step_memory', 64)
        )
        
        # Performance targets
        self.performance_targets = config.get('performance', {})
        
        # Cost settings
        self.cost_weights = config.get('costs', {})
        
        # Optimization history
        self.optimization_history: List[Tuple[datetime, OptimizationResult]] = []
        
        # Initialize state
        self.current_state = {
            'cpu_usage': [],
            'memory_usage': [],
            'response_times': [],
            'request_rates': [],
            'error_rates': []
        }
        
        # Start background optimization if enabled
        if config.get('auto_optimize', {}).get('enabled', True):
            asyncio.create_task(self._auto_optimization_loop())

    async def optimize_resources(
        self,
        current_metrics: Dict[str, float],
        predicted_workload: Optional[List[float]] = None,
        strategy: Optional[OptimizationStrategy] = None
    ) -> OptimizationResult:
        """
        Optimize resource allocation based on current metrics and predictions.
        
        Args:
            current_metrics: Current system metrics
            predicted_workload: Optional workload predictions
            strategy: Optional optimization strategy override
            
        Returns:
            Optimization result with resource recommendations
        """
        try:
            # Update current state
            self._update_state(current_metrics)
            
            # Use specified strategy or default
            active_strategy = strategy or self.strategy
            
            # Get optimization parameters based on strategy
            params = self._get_strategy_parameters(active_strategy)
            
            # Perform optimization
            result = await self._optimize_resources(
                current_metrics,
                predicted_workload,
                params
            )
            
            # Record optimization result
            self.optimization_history.append((datetime.utcnow(), result))
            
            # Trim history if needed
            max_history = self.config.get('max_history_size', 1000)
            if len(self.optimization_history) > max_history:
                self.optimization_history = self.optimization_history[-max_history:]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error optimizing resources: {str(e)}")
            raise

    async def _optimize_resources(
        self,
        current_metrics: Dict[str, float],
        predicted_workload: Optional[List[float]],
        strategy_params: Dict
    ) -> OptimizationResult:
        """Perform resource optimization."""
        # Get current usage
        current_cpu = current_metrics.get('cpu_usage', 0)
        current_memory = current_metrics.get('memory_usage', 0)
        
        # Define optimization bounds
        bounds = [
            (self.resource_limits.min_cpu, self.resource_limits.max_cpu),
            (self.resource_limits.min_memory, self.resource_limits.max_memory)
        ]
        
        # Initial guess (current allocation)
        x0 = [current_cpu, current_memory]
        
        # Perform optimization
        result = minimize(
            lambda x: self._objective_function(
                x,
                current_metrics,
                predicted_workload,
                strategy_params
            ),
            x0,
            bounds=bounds,
            method='SLSQP',
            constraints=self._get_constraints(current_metrics)
        )
        
        if not result.success:
            self.logger.warning(f"Optimization did not converge: {result.message}")
        
        # Calculate metrics for the result
        confidence = self._calculate_confidence(result.x, current_metrics)
        performance = self._predict_performance(result.x, current_metrics)
        savings = self._calculate_cost_savings(result.x, [current_cpu, current_memory])
        risk = self._calculate_risk(result.x, current_metrics, predicted_workload)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            result.x,
            [current_cpu, current_memory],
            current_metrics,
            predicted_workload
        )
        
        return OptimizationResult(
            cpu_allocation=result.x[0],
            memory_allocation=result.x[1],
            confidence=confidence,
            predicted_performance=performance,
            cost_savings=savings,
            risk_level=risk,
            recommendations=recommendations
        )

    def _objective_function(
        self,
        x: List[float],
        current_metrics: Dict[str, float],
        predicted_workload: Optional[List[float]],
        strategy_params: Dict
    ) -> float:
        """
        Objective function for optimization.
        Combines performance, cost, and risk metrics.
        """
        cpu, memory = x
        
        # Performance cost
        perf_cost = self._performance_cost(cpu, memory, current_metrics)
        
        # Resource cost
        resource_cost = self._resource_cost(cpu, memory)
        
        # Risk cost
        risk_cost = self._risk_cost(
            cpu,
            memory,
            current_metrics,
            predicted_workload
        )
        
        # Combine costs based on strategy weights
        total_cost = (
            strategy_params['performance_weight'] * perf_cost +
            strategy_params['resource_weight'] * resource_cost +
            strategy_params['risk_weight'] * risk_cost
        )
        
        return total_cost

    def _performance_cost(
        self,
        cpu: float,
        memory: float,
        metrics: Dict[str, float]
    ) -> float:
        """Calculate performance cost component."""
        # Estimate response time based on resource allocation
        est_response_time = self._estimate_response_time(cpu, memory, metrics)
        
        # Get target response time
        target_response_time = self.performance_targets.get('response_time', 100)
        
        # Calculate cost based on difference from target
        return max(0, est_response_time - target_response_time) ** 2

    def _resource_cost(
        self,
        cpu: float,
        memory: float
    ) -> float:
        """Calculate resource cost component."""
        cpu_cost = cpu * self.cost_weights.get('cpu_cost', 1.0)
        memory_cost = memory * self.cost_weights.get('memory_cost', 1.0)
        return cpu_cost + memory_cost

    def _risk_cost(
        self,
        cpu: float,
        memory: float,
        metrics: Dict[str, float],
        predicted_workload: Optional[List[float]]
    ) -> float:
        """Calculate risk cost component."""
        # Base risk on resource headroom
        cpu_headroom = (cpu - metrics.get('cpu_usage', 0)) / cpu
        memory_headroom = (memory - metrics.get('memory_usage', 0)) / memory
        
        # Include prediction risk if available
        if predicted_workload:
            max_predicted = max(predicted_workload)
            prediction_risk = max(0, max_predicted - cpu)
        else:
            prediction_risk = 0
        
        return -min(cpu_headroom, memory_headroom) + prediction_risk

    def _get_constraints(
        self,
        current_metrics: Dict[str, float]
    ) -> List[Dict]:
        """Generate optimization constraints."""
        constraints = []
        
        # Minimum resource headroom constraint
        min_headroom = self.config.get('constraints', {}).get('min_headroom', 0.2)
        
        constraints.append({
            'type': 'ineq',
            'fun': lambda x: x[0] - current_metrics.get('cpu_usage', 0) - min_headroom
        })
        
        constraints.append({
            'type': 'ineq',
            'fun': lambda x: x[1] - current_metrics.get('memory_usage', 0) - min_headroom
        })
        
        return constraints

    def _estimate_response_time(
        self,
        cpu: float,
        memory: float,
        metrics: Dict[str, float]
    ) -> float:
        """Estimate response time based on resource allocation."""
        # Simple response time model
        base_response_time = metrics.get('response_time', 100)
        cpu_factor = metrics.get('cpu_usage', 0) / cpu if cpu > 0 else float('inf')
        memory_factor = metrics.get('memory_usage', 0) / memory if memory > 0 else float('inf')
        
        return base_response_time * max(cpu_factor, memory_factor)

    def _calculate_confidence(
        self,
        allocation: List[float],
        metrics: Dict[str, float]
    ) -> float:
        """Calculate confidence in optimization result."""
        # Base confidence on resource headroom and historical accuracy
        cpu_headroom = (allocation[0] - metrics.get('cpu_usage', 0)) / allocation[0]
        memory_headroom = (allocation[1] - metrics.get('memory_usage', 0)) / allocation[1]
        
        # Historical accuracy factor
        if self.optimization_history:
            recent_accuracy = self._calculate_historical_accuracy()
        else:
            recent_accuracy = 1.0
        
        return min(1.0, min(cpu_headroom, memory_headroom) * recent_accuracy)

    def _predict_performance(
        self,
        allocation: List[float],
        metrics: Dict[str, float]
    ) -> float:
        """Predict performance with given allocation."""
        # Estimate response time
        est_response_time = self._estimate_response_time(
            allocation[0],
            allocation[1],
            metrics
        )
        
        # Convert to performance score (0-1)
        target_response_time = self.performance_targets.get('response_time', 100)
        return max(0, 1 - (est_response_time / target_response_time))

    def _calculate_cost_savings(
        self,
        new_allocation: List[float],
        current_allocation: List[float]
    ) -> float:
        """Calculate potential cost savings."""
        current_cost = self._resource_cost(
            current_allocation[0],
            current_allocation[1]
        )
        new_cost = self._resource_cost(
            new_allocation[0],
            new_allocation[1]
        )
        return max(0, current_cost - new_cost)

    def _calculate_risk(
        self,
        allocation: List[float],
        metrics: Dict[str, float],
        predicted_workload: Optional[List[float]]
    ) -> float:
        """Calculate risk level of allocation."""
        # Base risk calculation
        base_risk = self._risk_cost(
            allocation[0],
            allocation[1],
            metrics,
            predicted_workload
        )
        
        # Scale to 0-1 range
        return min(1.0, max(0.0, base_risk))

    def _generate_recommendations(
        self,
        new_allocation: List[float],
        current_allocation: List[float],
        metrics: Dict[str, float],
        predicted_workload: Optional[List[float]]
    ) -> List[str]:
        """Generate human-readable recommendations."""
        recommendations = []
        
        # CPU recommendations
        cpu_diff = new_allocation[0] - current_allocation[0]
        if abs(cpu_diff) > self.resource_limits.step_cpu:
            action = "increase" if cpu_diff > 0 else "decrease"
            recommendations.append(
                f"Recommend {action} CPU allocation by {abs(cpu_diff):.1f} cores"
            )
        
        # Memory recommendations
        memory_diff = new_allocation[1] - current_allocation[1]
        if abs(memory_diff) > self.resource_limits.step_memory:
            action = "increase" if memory_diff > 0 else "decrease"
            recommendations.append(
                f"Recommend {action} memory allocation by {abs(memory_diff):.0f} MB"
            )
        
        # Additional insights
        if predicted_workload:
            max_predicted = max(predicted_workload)
            if max_predicted > new_allocation[0]:
                recommendations.append(
                    "Warning: Predicted workload exceeds recommended CPU allocation"
                )
        
        return recommendations

    def _get_strategy_parameters(
        self,
        strategy: OptimizationStrategy
    ) -> Dict[str, float]:
        """Get optimization parameters for given strategy."""
        if strategy == OptimizationStrategy.CONSERVATIVE:
            return {
                'performance_weight': 0.4,
                'resource_weight': 0.2,
                'risk_weight': 0.4
            }
        elif strategy == OptimizationStrategy.AGGRESSIVE:
            return {
                'performance_weight': 0.6,
                'resource_weight': 0.3,
                'risk_weight': 0.1
            }
        elif strategy == OptimizationStrategy.COST_FOCUSED:
            return {
                'performance_weight': 0.2,
                'resource_weight': 0.6,
                'risk_weight': 0.2
            }
        elif strategy == OptimizationStrategy.PERFORMANCE_FOCUSED:
            return {
                'performance_weight': 0.7,
                'resource_weight': 0.1,
                'risk_weight': 0.2
            }
        else:  # BALANCED
            return {
                'performance_weight': 0.4,
                'resource_weight': 0.3,
                'risk_weight': 0.3
            }

    def _update_state(self, metrics: Dict[str, float]):
        """Update current state with new metrics."""
        for metric, value in metrics.items():
            if metric in self.current_state:
                self.current_state[metric].append(value)
                
                # Keep limited history
                max_history = self.config.get('state_history_size', 1000)
                if len(self.current_state[metric]) > max_history:
                    self.current_state[metric] = self.current_state[metric][-max_history:]

    def _calculate_historical_accuracy(self) -> float:
        """Calculate historical optimization accuracy."""
        if not self.optimization_history:
            return 1.0
            
        recent_results = self.optimization_history[-10:]  # Last 10 optimizations
        accuracy_scores = []
        
        for _, result in recent_results:
            # Compare predicted performance with actual
            accuracy_scores.append(result.confidence)
            
        return np.mean(accuracy_scores) if accuracy_scores else 1.0

    async def _auto_optimization_loop(self):
        """Background task for automatic optimization."""
        while True:
            try:
                # Get current metrics
                current_metrics = await self._get_current_metrics()
                
                # Perform optimization
                result = await self.optimize_resources(current_metrics)
                
                # Log results
                self.logger.info(
                    f"Auto-optimization result: CPU={result.cpu_allocation:.1f}, "
                    f"Memory={result.memory_allocation:.0f}, "
                    f"Confidence={result.confidence:.2f}"
                )
                
                # Sleep until next optimization
                interval = self.config.get('auto_optimize', {}).get('interval', 300)
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in auto-optimization loop: {str(e)}")
                await asyncio.sleep(60)

    async def _get_current_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        # This should be implemented based on your metrics collection system
        metrics = {}
        
        for metric in self.current_state:
            values = self.current_state[metric]
            if values:
                metrics[metric] = values[-1]
            else:
                metrics[metric] = 0.0
                
        return metrics

# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'strategy': 'balanced',
        'resources': {
            'min_cpu': 0.1,
            'max_cpu': 4.0,
            'min_memory': 128,
            'max_memory': 8192,
            'step_cpu': 0.1,
            'step_memory': 64
        },
        'performance': {
            'response_time': 100
        },
        'costs': {
            'cpu_cost': 1.0,
            'memory_cost': 0.1
        },
        'auto_optimize': {
            'enabled': True,
            'interval': 300
        }
    }
    
    # Initialize optimizer
    optimizer = ResourceOptimizer(config)
    
    # Example optimization
    async def main():
        current_metrics = {
            'cpu_usage': 1.5,
            'memory_usage': 2048,
            'response_time': 150,
            'request_rate': 100,
            'error_rate': 0.01
        }
        
        predicted_workload = [1.8, 2.0, 1.7, 1.9]
        
        result = await optimizer.optimize_resources(
            current_metrics,
            predicted_workload,
            OptimizationStrategy.BALANCED
        )
        
        print(f"Optimized CPU: {result.cpu_allocation:.1f}")
        print(f"Optimized Memory: {result.memory_allocation:.0f}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Predicted Performance: {result.predicted_performance:.2f}")
        print(f"Cost Savings: ${result.cost_savings:.2f}")
        print(f"Risk Level: {result.risk_level:.2f}")
        print("\nRecommendations:")
        for rec in result.recommendations:
            print(f"- {rec}")
    
    # Run example
    asyncio.run(main())