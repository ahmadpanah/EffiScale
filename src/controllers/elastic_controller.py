from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio
import logging
import docker
from enum import Enum
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class ScalingAction(Enum):
    """Types of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    NO_ACTION = "no_action"

@dataclass
class ScalingDecision:
    """Container for scaling decisions."""
    action: ScalingAction
    magnitude: float
    confidence: float
    reason: str
    target_containers: int
    target_cpu: float
    target_memory: float
    estimated_impact: Dict[str, float]

class ElasticController:
    """
    Controls elastic scaling of containers based on workload,
    performance metrics, and resource optimization recommendations.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the elastic controller.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize Docker client
        self.docker_client = docker.from_env()
        
        # Initialize components
        self.min_instances = config.get('scaling', {}).get('min_instances', 1)
        self.max_instances = config.get('scaling', {}).get('max_instances', 10)
        self.scale_up_threshold = config.get('scaling', {}).get('scale_up_threshold', 0.75)
        self.scale_down_threshold = config.get('scaling', {}).get('scale_down_threshold', 0.25)
        self.cooldown_period = config.get('scaling', {}).get('cooldown_period', 300)
        
        # Scaling history
        self.scaling_history: List[Tuple[datetime, ScalingDecision]] = []
        self.last_scaling_time: Optional[datetime] = None
        
        # Resource settings
        self.resource_settings = config.get('resources', {})
        
        # Performance targets
        self.performance_targets = config.get('performance', {})
        
        # Initialize state
        self.current_state = {
            'instances': {},
            'metrics': {},
            'scaling_locks': set()
        }
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(
            max_workers=config.get('max_workers', 5)
        )
        
        # Start background monitoring
        if config.get('auto_scale', {}).get('enabled', True):
            asyncio.create_task(self._monitoring_loop())

    async def evaluate_scaling(
        self,
        current_metrics: Dict[str, float],
        predicted_workload: Optional[List[float]] = None,
        optimization_result: Optional[Dict] = None
    ) -> ScalingDecision:
        """
        Evaluate scaling needs based on current metrics and predictions.
        
        Args:
            current_metrics: Current system metrics
            predicted_workload: Optional workload predictions
            optimization_result: Optional resource optimization recommendations
            
        Returns:
            Scaling decision
        """
        try:
            # Check cooldown period
            if self.last_scaling_time and \
               (datetime.utcnow() - self.last_scaling_time).total_seconds() < self.cooldown_period:
                return ScalingDecision(
                    action=ScalingAction.NO_ACTION,
                    magnitude=0.0,
                    confidence=1.0,
                    reason="In cooldown period",
                    target_containers=self._get_current_instance_count(),
                    target_cpu=self._get_current_cpu_allocation(),
                    target_memory=self._get_current_memory_allocation(),
                    estimated_impact={}
                )
            
            # Evaluate horizontal scaling
            horizontal_decision = await self._evaluate_horizontal_scaling(
                current_metrics,
                predicted_workload
            )
            
            # Evaluate vertical scaling
            vertical_decision = await self._evaluate_vertical_scaling(
                current_metrics,
                optimization_result
            )
            
            # Choose final decision
            final_decision = self._combine_scaling_decisions(
                horizontal_decision,
                vertical_decision
            )
            
            # Update history
            self.scaling_history.append((datetime.utcnow(), final_decision))
            
            # Trim history if needed
            max_history = self.config.get('max_history_size', 1000)
            if len(self.scaling_history) > max_history:
                self.scaling_history = self.scaling_history[-max_history:]
            
            return final_decision
            
        except Exception as e:
            self.logger.error(f"Error evaluating scaling: {str(e)}")
            raise

    async def apply_scaling_decision(
        self,
        decision: ScalingDecision
    ) -> bool:
        """
        Apply a scaling decision.
        
        Args:
            decision: Scaling decision to apply
            
        Returns:
            Success status
        """
        try:
            if decision.action == ScalingAction.NO_ACTION:
                return True
            
            # Check for scaling locks
            service_name = self.config.get('service_name')
            if service_name in self.current_state['scaling_locks']:
                self.logger.warning(f"Scaling locked for service {service_name}")
                return False
            
            success = False
            
            if decision.action in [ScalingAction.SCALE_OUT, ScalingAction.SCALE_IN]:
                success = await self._apply_horizontal_scaling(decision)
            else:
                success = await self._apply_vertical_scaling(decision)
            
            if success:
                self.last_scaling_time = datetime.utcnow()
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error applying scaling decision: {str(e)}")
            return False

    async def _evaluate_horizontal_scaling(
        self,
        metrics: Dict[str, float],
        predicted_workload: Optional[List[float]]
    ) -> ScalingDecision:
        """Evaluate need for horizontal scaling."""
        current_instances = self._get_current_instance_count()
        cpu_usage = metrics.get('cpu_usage', 0)
        memory_usage = metrics.get('memory_usage', 0)
        request_rate = metrics.get('request_rate', 0)
        
        # Calculate scaling factors
        cpu_factor = cpu_usage / self.scale_up_threshold if self.scale_up_threshold > 0 else float('inf')
        memory_factor = memory_usage / self.scale_up_threshold if self.scale_up_threshold > 0 else float('inf')
        
        # Consider predictions if available
        if predicted_workload:
            predicted_factor = max(predicted_workload) / self.scale_up_threshold
            scaling_factor = max(cpu_factor, memory_factor, predicted_factor)
        else:
            scaling_factor = max(cpu_factor, memory_factor)
        
        # Determine scaling action
        if scaling_factor > 1.0 and current_instances < self.max_instances:
            # Scale out
            additional_instances = min(
                int(np.ceil(current_instances * (scaling_factor - 1))),
                self.max_instances - current_instances
            )
            
            return ScalingDecision(
                action=ScalingAction.SCALE_OUT,
                magnitude=additional_instances,
                confidence=min(scaling_factor - 1, 1.0),
                reason=f"High resource utilization (factor: {scaling_factor:.2f})",
                target_containers=current_instances + additional_instances,
                target_cpu=self._get_current_cpu_allocation(),
                target_memory=self._get_current_memory_allocation(),
                estimated_impact={
                    'cpu_usage': cpu_usage / (current_instances + additional_instances),
                    'response_time': metrics.get('response_time', 0) * 0.8,
                    'cost': self._estimate_cost_impact(additional_instances)
                }
            )
            
        elif scaling_factor < self.scale_down_threshold and current_instances > self.min_instances:
            # Scale in
            reduce_instances = min(
                int(np.ceil(current_instances * (1 - scaling_factor/self.scale_down_threshold))),
                current_instances - self.min_instances
            )
            
            return ScalingDecision(
                action=ScalingAction.SCALE_IN,
                magnitude=reduce_instances,
                confidence=min(1 - scaling_factor/self.scale_down_threshold, 1.0),
                reason=f"Low resource utilization (factor: {scaling_factor:.2f})",
                target_containers=current_instances - reduce_instances,
                target_cpu=self._get_current_cpu_allocation(),
                target_memory=self._get_current_memory_allocation(),
                estimated_impact={
                    'cpu_usage': cpu_usage / (current_instances - reduce_instances),
                    'response_time': metrics.get('response_time', 0) * 1.2,
                    'cost': -self._estimate_cost_impact(reduce_instances)
                }
            )
        
        return ScalingDecision(
            action=ScalingAction.NO_ACTION,
            magnitude=0.0,
            confidence=1.0,
            reason="No horizontal scaling needed",
            target_containers=current_instances,
            target_cpu=self._get_current_cpu_allocation(),
            target_memory=self._get_current_memory_allocation(),
            estimated_impact={}
        )

    async def _evaluate_vertical_scaling(
        self,
        metrics: Dict[str, float],
        optimization_result: Optional[Dict]
    ) -> ScalingDecision:
        """Evaluate need for vertical scaling."""
        current_cpu = self._get_current_cpu_allocation()
        current_memory = self._get_current_memory_allocation()
        
        if optimization_result:
            # Use optimization recommendations
            target_cpu = optimization_result.get('cpu_allocation', current_cpu)
            target_memory = optimization_result.get('memory_allocation', current_memory)
            confidence = optimization_result.get('confidence', 0.5)
            
            if abs(target_cpu - current_cpu) > self.resource_settings.get('cpu_step', 0.1) or \
               abs(target_memory - current_memory) > self.resource_settings.get('memory_step', 64):
                
                action = ScalingAction.SCALE_UP if target_cpu > current_cpu else ScalingAction.SCALE_DOWN
                magnitude = max(
                    abs(target_cpu - current_cpu) / current_cpu,
                    abs(target_memory - current_memory) / current_memory
                )
                
                return ScalingDecision(
                    action=action,
                    magnitude=magnitude,
                    confidence=confidence,
                    reason="Resource optimization recommendation",
                    target_containers=self._get_current_instance_count(),
                    target_cpu=target_cpu,
                    target_memory=target_memory,
                    estimated_impact=optimization_result.get('estimated_impact', {})
                )
        
        return ScalingDecision(
            action=ScalingAction.NO_ACTION,
            magnitude=0.0,
            confidence=1.0,
            reason="No vertical scaling needed",
            target_containers=self._get_current_instance_count(),
            target_cpu=current_cpu,
            target_memory=current_memory,
            estimated_impact={}
        )

    def _combine_scaling_decisions(
        self,
        horizontal: ScalingDecision,
        vertical: ScalingDecision
    ) -> ScalingDecision:
        """Combine horizontal and vertical scaling decisions."""
        # Prefer horizontal scaling if both are recommended
        if horizontal.action != ScalingAction.NO_ACTION:
            return horizontal
        
        return vertical

    async def _apply_horizontal_scaling(
        self,
        decision: ScalingDecision
    ) -> bool:
        """Apply horizontal scaling decision."""
        try:
            service_name = self.config.get('service_name')
            service = self.docker_client.services.get(service_name)
            
            current_replicas = service.attrs['Spec']['Mode']['Replicated']['Replicas']
            
            if decision.action == ScalingAction.SCALE_OUT:
                target_replicas = current_replicas + int(decision.magnitude)
            else:  # SCALE_IN
                target_replicas = current_replicas - int(decision.magnitude)
            
            # Update service
            service.update(
                mode={'Replicated': {'Replicas': target_replicas}},
                update_config={
                    'Parallelism': 1,
                    'Delay': 10,
                    'Order': 'start-first'
                }
            )
            
            self.logger.info(
                f"Scaled service {service_name} to {target_replicas} replicas"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying horizontal scaling: {str(e)}")
            return False

    async def _apply_vertical_scaling(
        self,
        decision: ScalingDecision
    ) -> bool:
        """Apply vertical scaling decision."""
        try:
            service_name = self.config.get('service_name')
            service = self.docker_client.services.get(service_name)
            
            # Prepare resource updates
            resources = service.attrs['Spec']['TaskTemplate']['Resources']
            
            # Update CPU limits
            if 'NanoCPUs' in resources['Limits']:
                resources['Limits']['NanoCPUs'] = int(decision.target_cpu * 1e9)
            
            # Update Memory limits
            if 'MemoryBytes' in resources['Limits']:
                resources['Limits']['MemoryBytes'] = int(decision.target_memory * 1024 * 1024)
            
            # Update service
            service.update(
                task_template={
                    'Resources': resources
                }
            )
            
            self.logger.info(
                f"Updated resources for service {service_name}: "
                f"CPU={decision.target_cpu:.1f}, Memory={decision.target_memory:.0f}MB"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying vertical scaling: {str(e)}")
            return False

    def _get_current_instance_count(self) -> int:
        """Get current number of container instances."""
        try:
            service_name = self.config.get('service_name')
            service = self.docker_client.services.get(service_name)
            return service.attrs['Spec']['Mode']['Replicated']['Replicas']
        except Exception:
            return 0

    def _get_current_cpu_allocation(self) -> float:
        """Get current CPU allocation."""
        try:
            service_name = self.config.get('service_name')
            service = self.docker_client.services.get(service_name)
            nano_cpus = service.attrs['Spec']['TaskTemplate']['Resources']['Limits']['NanoCPUs']
            return nano_cpus / 1e9
        except Exception:
            return 0.0

    def _get_current_memory_allocation(self) -> float:
        """Get current memory allocation in MB."""
        try:
            service_name = self.config.get('service_name')
            service = self.docker_client.services.get(service_name)
            memory_bytes = service.attrs['Spec']['TaskTemplate']['Resources']['Limits']['MemoryBytes']
            return memory_bytes / (1024 * 1024)
        except Exception:
            return 0.0

    def _estimate_cost_impact(self, instance_change: int) -> float:
        """Estimate cost impact of scaling decision."""
        cpu_cost = self.config.get('costs', {}).get('cpu_cost', 1.0)
        memory_cost = self.config.get('costs', {}).get('memory_cost', 0.1)
        
        cpu_allocation = self._get_current_cpu_allocation()
        memory_allocation = self._get_current_memory_allocation()
        
        return instance_change * (
            cpu_allocation * cpu_cost +
            memory_allocation * memory_cost
        )

    async def _monitoring_loop(self):
        """Background task for automatic scaling monitoring."""
        while True:
            try:
                # Get current metrics
                metrics = await self._get_current_metrics()
                
                # Evaluate scaling
                decision = await self.evaluate_scaling(metrics)
                
                # Apply decision if needed
                if decision.action != ScalingAction.NO_ACTION:
                    await self.apply_scaling_decision(decision)
                
                # Sleep until next check
                interval = self.config.get('auto_scale', {}).get('interval', 60)
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(60)

    async def _get_current_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        # This should be implemented based on your metrics collection system
        return self.current_state.get('metrics', {})

    def update_metrics(self, metrics: Dict[str, float]):
        """Update current metrics state."""
        self.current_state['metrics'].update(metrics)

    def lock_scaling(self, service_name: str):
        """Lock scaling for a service."""
        self.current_state['scaling_locks'].add(service_name)

    def unlock_scaling(self, service_name: str):
        """Unlock scaling for a service."""
        self.current_state['scaling_locks'].discard(service_name)

# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'service_name': 'example-service',
        'scaling': {
            'min_instances': 1,
            'max_instances': 10,
            'scale_up_threshold': 0.75,
            'scale_down_threshold': 0.25,
            'cooldown_period': 300
        },
        'resources': {
            'cpu_step': 0.1,
            'memory_step': 64
        },
        'performance': {
            'target_response_time': 100
        },
        'costs': {
            'cpu_cost': 1.0,
            'memory_cost': 0.1
        },
        'auto_scale': {
            'enabled': True,
            'interval': 60
        }
    }
    
    # Initialize controller
    controller = ElasticController(config)
    
    # Example scaling evaluation
    async def main():
        # Current metrics
        metrics = {
            'cpu_usage': 0.8,
            'memory_usage': 2048,
            'response_time': 150,
            'request_rate': 100
        }
        
        # Predicted workload
        predicted_workload = [0.85, 0.9, 0.82, 0.88]
        
        # Resource optimization result
        optimization_result = {
            'cpu_allocation': 2.0,
            'memory_allocation': 4096,
            'confidence': 0.8,
            'estimated_impact': {
                'response_time': 100,
                'cost': 10.0
            }
        }
        
        # Evaluate scaling
        decision = await controller.evaluate_scaling(
            metrics,
            predicted_workload,
            optimization_result
        )
        
        print(f"Scaling Decision:")
        print(f"Action: {decision.action.value}")
        print(f"Magnitude: {decision.magnitude:.2f}")
        print(f"Confidence: {decision.confidence:.2f}")
        print(f"Reason: {decision.reason}")
        print(f"Target Containers: {decision.target_containers}")
        print(f"Target CPU: {decision.target_cpu:.1f}")
        print(f"Target Memory: {decision.target_memory:.0f}MB")
        print("Estimated Impact:", json.dumps(decision.estimated_impact, indent=2))
        
        # Apply decision
        if decision.action != ScalingAction.NO_ACTION:
            success = await controller.apply_scaling_decision(decision)
            print(f"Scaling applied: {success}")
    
    # Run example
    asyncio.run(main())