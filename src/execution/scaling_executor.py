from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import asyncio
import logging
import json
from enum import Enum
import hashlib
from collections import defaultdict
import aioredis
import kubernetes.client as k8s
import kubernetes.client.rest
from kubernetes.client.rest import ApiException
import boto3
import aioboto3
import aiohttp

class ScalingAction(Enum):
    """Types of scaling actions."""
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MIGRATE = "migrate"
    REBALANCE = "rebalance"

class ResourceType(Enum):
    """Types of resources to scale."""
    KUBERNETES_DEPLOYMENT = "k8s_deployment"
    KUBERNETES_STATEFULSET = "k8s_statefulset"
    CUSTOM_SERVICE = "custom_service"

class ScalingStatus(Enum):
    """Status of scaling operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"

@dataclass
class ScalingOperation:
    """Container for scaling operations."""
    operation_id: str
    action: ScalingAction
    resource_type: ResourceType
    resource_id: str
    target_state: Dict
    current_state: Dict
    status: ScalingStatus
    metadata: Dict
    timestamp: datetime = field(default_factory=datetime.utcnow)
    completion_time: Optional[datetime] = None
    error: Optional[str] = None
    metrics: Dict = field(default_factory=dict)
    rollback_info: Optional[Dict] = None

class ScalingExecutor:
    """
    Executes scaling decisions across different platforms and resources.
    Implements scaling operations with monitoring and error handling.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the scaling executor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.operations: Dict[str, ScalingOperation] = {}
        self.pending_operations: List[str] = []
        self.active_operations: List[str] = []
        
        # Platform clients
        self.k8s_client = None
        self.redis_pool = None
        
        # Operation limits
        self.max_concurrent_operations = config.get('max_concurrent_operations', 5)
        self.operation_timeout = config.get('operation_timeout', 300)  # seconds
        
        # Monitoring settings
        self.monitoring_interval = config.get('monitoring_interval', 10)  # seconds
        self.metric_collection_interval = config.get('metric_collection_interval', 60)
        
        # Start background tasks
        asyncio.create_task(self._initialize_clients())
        asyncio.create_task(self._operation_processor_loop())
        asyncio.create_task(self._monitoring_loop())
        asyncio.create_task(self._metric_collection_loop())

    async def execute_scaling(
        self,
        action: ScalingAction,
        resource_type: ResourceType,
        resource_id: str,
        target_state: Dict,
        metadata: Optional[Dict] = None
    ) -> ScalingOperation:
        """
        Execute scaling operation.
        
        Args:
            action: Scaling action to perform
            resource_type: Type of resource to scale
            resource_id: Resource identifier
            target_state: Desired state after scaling
            metadata: Optional metadata
            
        Returns:
            ScalingOperation object
        """
        try:
            # Generate operation ID
            operation_id = self._generate_operation_id(
                action,
                resource_type,
                resource_id
            )
            
            # Get current state
            current_state = await self._get_current_state(
                resource_type,
                resource_id
            )
            
            # Create operation
            operation = ScalingOperation(
                operation_id=operation_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                target_state=target_state,
                current_state=current_state,
                status=ScalingStatus.PENDING,
                metadata=metadata or {},
                rollback_info={
                    'initial_state': current_state,
                    'checkpoints': []
                }
            )
            
            # Store operation
            self.operations[operation_id] = operation
            self.pending_operations.append(operation_id)
            
            # Broadcast operation start
            await self._broadcast_operation_status(operation)
            
            return operation
            
        except Exception as e:
            self.logger.error(f"Error executing scaling: {str(e)}")
            raise

    async def _get_current_state(
        self,
        resource_type: ResourceType,
        resource_id: str
    ) -> Dict:
        """Get current state of resource."""
        try:
            if resource_type == ResourceType.KUBERNETES_DEPLOYMENT:
                return await self._get_k8s_deployment_state(resource_id)
            elif resource_type == ResourceType.KUBERNETES_STATEFULSET:
                return await self._get_k8s_statefulset_state(resource_id)
            elif resource_type == ResourceType.CUSTOM_SERVICE:
                return await self._get_custom_service_state(resource_id)
            else:
                raise ValueError(f"Unsupported resource type: {resource_type}")
            
        except Exception as e:
            self.logger.error(f"Error getting current state: {str(e)}")
            return {}

    async def _process_operation(
        self,
        operation: ScalingOperation
    ):
        """Process scaling operation."""
        try:
            operation.status = ScalingStatus.IN_PROGRESS
            await self._broadcast_operation_status(operation)
            
            # Execute scaling based on resource type
            if operation.resource_type == ResourceType.KUBERNETES_DEPLOYMENT:
                success = await self._scale_k8s_deployment(operation)
            elif operation.resource_type == ResourceType.KUBERNETES_STATEFULSET:
                success = await self._scale_k8s_statefulset(operation)
            elif operation.resource_type == ResourceType.CUSTOM_SERVICE:
                success = await self._scale_custom_service(operation)
            else:
                raise ValueError(f"Unsupported resource type: {operation.resource_type}")
            
            if success:
                operation.status = ScalingStatus.COMPLETED
                operation.completion_time = datetime.utcnow()
            else:
                operation.status = ScalingStatus.FAILED
                await self._rollback_operation(operation)
            
            await self._broadcast_operation_status(operation)
            
        except Exception as e:
            operation.status = ScalingStatus.FAILED
            operation.error = str(e)
            await self._rollback_operation(operation)
            await self._broadcast_operation_status(operation)

    async def _scale_k8s_deployment(
        self,
        operation: ScalingOperation
    ) -> bool:
        """Scale Kubernetes deployment."""
        try:
            if not self.k8s_client:
                raise ValueError("Kubernetes client not initialized")
            
            namespace = operation.metadata.get('namespace', 'default')
            deployment_name = operation.resource_id
            
            # Get deployment
            deployment = self.k8s_client.AppsV1Api().read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            
            # Update replicas
            if operation.action in [ScalingAction.SCALE_OUT, ScalingAction.SCALE_IN]:
                deployment.spec.replicas = operation.target_state.get('replicas')
            
            # Update resources if scaling up/down
            if operation.action in [ScalingAction.SCALE_UP, ScalingAction.SCALE_DOWN]:
                containers = deployment.spec.template.spec.containers
                for container in containers:
                    if container.name == operation.metadata.get('container_name'):
                        container.resources = k8s.V1ResourceRequirements(
                            requests=operation.target_state.get('resources', {}).get('requests', {}),
                            limits=operation.target_state.get('resources', {}).get('limits', {})
                        )
            
            # Apply update
            self.k8s_client.AppsV1Api().patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body=deployment
            )
            
            # Wait for rollout
            return await self._wait_for_k8s_deployment_rollout(
                deployment_name,
                namespace,
                operation
            )
            
        except ApiException as e:
            self.logger.error(f"Kubernetes API error: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Error scaling K8s deployment: {str(e)}")
            return False
   

    async def _scale_custom_service(
        self,
        operation: ScalingOperation
    ) -> bool:
        """Scale custom service using provided API."""
        try:
            api_config = self.config.get('custom_service_api', {})
            if not api_config:
                raise ValueError("Custom service API not configured")
            
            async with aiohttp.ClientSession() as session:
                # Prepare request
                url = f"{api_config['base_url']}/scale/{operation.resource_id}"
                headers = {
                    'Authorization': f"Bearer {api_config['api_key']}",
                    'Content-Type': 'application/json'
                }
                
                payload = {
                    'action': operation.action.value,
                    'target_state': operation.target_state,
                    'metadata': operation.metadata
                }
                
                # Send scaling request
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status != 200:
                        raise ValueError(f"API error: {await response.text()}")
                    
                    # Wait for operation completion
                    return await self._wait_for_custom_service_operation(
                        operation,
                        session,
                        api_config
                    )
            
        except Exception as e:
            self.logger.error(f"Error scaling custom service: {str(e)}")
            return False

    async def _wait_for_k8s_deployment_rollout(
        self,
        deployment_name: str,
        namespace: str,
        operation: ScalingOperation
    ) -> bool:
        """Wait for Kubernetes deployment rollout completion."""
        start_time = datetime.utcnow()
        while True:
            try:
                deployment = self.k8s_client.AppsV1Api().read_namespaced_deployment(
                    name=deployment_name,
                    namespace=namespace
                )
                
                # Check rollout status
                if deployment.status.updated_replicas == deployment.spec.replicas and \
                   deployment.status.available_replicas == deployment.spec.replicas:
                    return True
                
                # Check timeout
                if (datetime.utcnow() - start_time).total_seconds() > self.operation_timeout:
                    operation.error = "Rollout timeout exceeded"
                    return False
                
                # Store checkpoint
                operation.rollback_info['checkpoints'].append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'status': deployment.status.to_dict()
                })
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                operation.error = f"Error monitoring rollout: {str(e)}"
                return False

    

    async def _rollback_operation(
        self,
        operation: ScalingOperation
    ):
        """Roll back failed operation."""
        try:
            initial_state = operation.rollback_info.get('initial_state')
            if not initial_state:
                return
            
            self.logger.info(f"Rolling back operation {operation.operation_id}")
            
            # Create rollback operation
            rollback_operation = ScalingOperation(
                operation_id=f"rollback_{operation.operation_id}",
                action=operation.action,
                resource_type=operation.resource_type,
                resource_id=operation.resource_id,
                target_state=initial_state,
                current_state=await self._get_current_state(
                    operation.resource_type,
                    operation.resource_id
                ),
                status=ScalingStatus.IN_PROGRESS,
                metadata={'rollback_for': operation.operation_id}
            )
            
            # Execute rollback
            await self._process_operation(rollback_operation)
            
            if rollback_operation.status == ScalingStatus.COMPLETED:
                operation.status = ScalingStatus.ROLLED_BACK
            
        except Exception as e:
            self.logger.error(f"Error rolling back operation: {str(e)}")
            operation.error = f"Rollback failed: {str(e)}"

    async def _operation_processor_loop(self):
        """Background task for processing scaling operations."""
        while True:
            try:
                # Process pending operations
                while len(self.active_operations) < self.max_concurrent_operations and \
                      self.pending_operations:
                    operation_id = self.pending_operations.pop(0)
                    operation = self.operations.get(operation_id)
                    
                    if operation:
                        self.active_operations.append(operation_id)
                        asyncio.create_task(self._process_operation(operation))
                
                # Clean up completed operations
                self.active_operations = [
                    op_id for op_id in self.active_operations
                    if self.operations[op_id].status == ScalingStatus.IN_PROGRESS
                ]
                
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in operation processor: {str(e)}")
                await asyncio.sleep(5)

    async def _monitoring_loop(self):
        """Background task for monitoring active operations."""
        while True:
            try:
                for operation_id in self.active_operations:
                    operation = self.operations.get(operation_id)
                    if operation and operation.status == ScalingStatus.IN_PROGRESS:
                        # Check timeout
                        if (datetime.utcnow() - operation.timestamp).total_seconds() > \
                           self.operation_timeout:
                            operation.status = ScalingStatus.FAILED
                            operation.error = "Operation timeout exceeded"
                            await self._rollback_operation(operation)
                            await self._broadcast_operation_status(operation)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(5)

    async def _metric_collection_loop(self):
        """Background task for collecting operation metrics."""
        while True:
            try:
                # Collect metrics for active operations
                for operation_id in self.active_operations:
                    operation = self.operations.get(operation_id)
                    if operation and operation.status == ScalingStatus.IN_PROGRESS:
                        metrics = await self._collect_operation_metrics(operation)
                        operation.metrics[datetime.utcnow().isoformat()] = metrics
                
                await asyncio.sleep(self.metric_collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in metric collection: {str(e)}")
                await asyncio.sleep(60)

    async def _collect_operation_metrics(
        self,
        operation: ScalingOperation
    ) -> Dict:
        """Collect metrics for scaling operation."""
        try:
            metrics = {}
            
            if operation.resource_type == ResourceType.KUBERNETES_DEPLOYMENT:
                metrics = await self._collect_k8s_metrics(operation)
            elif operation.resource_type == ResourceType.CUSTOM_SERVICE:
                metrics = await self._collect_custom_service_metrics(operation)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {str(e)}")
            return {}

    async def _broadcast_operation_status(
        self,
        operation: ScalingOperation
    ):
        """Broadcast operation status updates."""
        try:
            if self.redis_pool:
                # Prepare status message
                message = {
                    'operation_id': operation.operation_id,
                    'status': operation.status.value,
                    'resource_type': operation.resource_type.value,
                    'resource_id': operation.resource_id,
                    'action': operation.action.value,
                    'timestamp': datetime.utcnow().isoformat(),
                    'error': operation.error
                }
                
                # Publish to Redis
                await self.redis_pool.publish(
                    'scaling_operations',
                    json.dumps(message)
                )
            
        except Exception as e:
            self.logger.error(f"Error broadcasting status: {str(e)}")

    def _generate_operation_id(
        self,
        action: ScalingAction,
        resource_type: ResourceType,
        resource_id: str
    ) -> str:
        """Generate unique operation ID."""
        data = f"{action.value}:{resource_type.value}:{resource_id}:{datetime.utcnow().isoformat()}"
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    async def _initialize_clients(self):
        """Initialize platform clients."""
        try:
            # Initialize Kubernetes client
            if 'kubernetes' in self.config:
                k8s.config.load_config()
                self.k8s_client = k8s.client

            # Initialize Redis
            if 'redis_url' in self.config:
                self.redis_pool = await aioredis.create_redis_pool(
                    self.config['redis_url'],
                    minsize=5,
                    maxsize=10
                )
            
        except Exception as e:
            self.logger.error(f"Error initializing clients: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'kubernetes': {
            'config_path': '~/.kube/config'
        },
        'redis_url': 'redis://localhost',
        'max_concurrent_operations': 5,
        'operation_timeout': 300,
        'monitoring_interval': 10,
        'metric_collection_interval': 60
    }
    
    # Initialize executor
    executor = ScalingExecutor(config)
    
    # Example scaling operation
    async def main():
        # Scale Kubernetes deployment
        operation = await executor.execute_scaling(
            action=ScalingAction.SCALE_OUT,
            resource_type=ResourceType.KUBERNETES_DEPLOYMENT,
            resource_id='my-deployment',
            target_state={
                'replicas': 5,
                'resources': {
                    'requests': {
                        'cpu': '200m',
                        'memory': '256Mi'
                    },
                    'limits': {
                        'cpu': '500m',
                        'memory': '512Mi'
                    }
                }
            },
            metadata={
                'namespace': 'default',
                'container_name': 'app'
            }
        )
        
        print(f"Started scaling operation: {operation.operation_id}")
        
        # Wait for completion
        while operation.status in [ScalingStatus.PENDING, ScalingStatus.IN_PROGRESS]:
            print(f"Operation status: {operation.status.value}")
            await asyncio.sleep(5)
        
        print(f"Final status: {operation.status.value}")
        if operation.error:
            print(f"Error: {operation.error}")
        
        # Print metrics
        print("\nOperation metrics:")
        for timestamp, metrics in operation.metrics.items():
            print(f"\nTimestamp: {timestamp}")
            print(json.dumps(metrics, indent=2))
    
    # Run example
    asyncio.run(main())