from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import asyncio
import logging
import json
from enum import Enum
import hashlib
import docker
from docker.errors import DockerException
import kubernetes.client as k8s
from kubernetes.client.rest import ApiException
import aioredis
import aiodocker
from yarl import URL
import aiohttp
import os
import tarfile
import io
import tempfile
from pathlib import Path

class ContainerState(Enum):
    """Container states."""
    PENDING = "pending"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    DELETED = "deleted"
    UNKNOWN = "unknown"

class ContainerPlatform(Enum):
    """Container platforms."""
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    CUSTOM = "custom"

@dataclass
class Container:
    """Container information container."""
    container_id: str
    name: str
    platform: ContainerPlatform
    image: str
    state: ContainerState
    config: Dict
    metadata: Dict
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None
    metrics: Dict = field(default_factory=dict)

class ContainerManager:
    """
    Manages container lifecycle and resources.
    Implements container operations across different platforms.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the container manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.containers: Dict[str, Container] = {}
        self.platform_clients = {}
        self.redis_pool = None
        
        # Container settings
        self.max_containers = config.get('max_containers', 100)
        self.container_memory_limit = config.get('container_memory_limit', '512m')
        self.container_cpu_limit = config.get('container_cpu_limit', '500m')
        
        # Monitoring settings
        self.monitoring_interval = config.get('monitoring_interval', 10)
        self.metrics_interval = config.get('metrics_interval', 30)
        
        # Cleanup settings
        self.cleanup_interval = config.get('cleanup_interval', 3600)
        self.container_retention_days = config.get('container_retention_days', 7)
        
        # Start background tasks
        asyncio.create_task(self._initialize_clients())
        asyncio.create_task(self._container_monitor_loop())
        asyncio.create_task(self._metrics_collector_loop())
        asyncio.create_task(self._cleanup_loop())

    async def create_container(
        self,
        name: str,
        image: str,
        platform: ContainerPlatform,
        config: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ) -> Container:
        """
        Create new container.
        
        Args:
            name: Container name
            image: Container image
            platform: Container platform
            config: Optional container configuration
            metadata: Optional metadata
            
        Returns:
            Container object
        """
        try:
            # Generate container ID
            container_id = self._generate_container_id(name, image, platform)
            
            # Check container limit
            if len(self.containers) >= self.max_containers:
                raise ValueError("Maximum container limit reached")
            
            # Create container configuration
            container_config = self._prepare_container_config(
                name,
                image,
                config or {}
            )
            
            # Create container based on platform
            if platform == ContainerPlatform.DOCKER:
                await self._create_docker_container(
                    container_id,
                    name,
                    image,
                    container_config
                )
            elif platform == ContainerPlatform.KUBERNETES:
                await self._create_kubernetes_container(
                    container_id,
                    name,
                    image,
                    container_config
                )
            elif platform == ContainerPlatform.CUSTOM:
                await self._create_custom_container(
                    container_id,
                    name,
                    image,
                    container_config
                )
            else:
                raise ValueError(f"Unsupported platform: {platform}")
            
            # Create container object
            container = Container(
                container_id=container_id,
                name=name,
                platform=platform,
                image=image,
                state=ContainerState.PENDING,
                config=container_config,
                metadata=metadata or {}
            )
            
            # Store container
            self.containers[container_id] = container
            
            # Broadcast container creation
            await self._broadcast_container_event(
                'container_created',
                container
            )
            
            return container
            
        except Exception as e:
            self.logger.error(f"Error creating container: {str(e)}")
            raise

    async def start_container(
        self,
        container_id: str
    ) -> bool:
        """Start container."""
        try:
            container = self.containers.get(container_id)
            if not container:
                raise ValueError(f"Container not found: {container_id}")
            
            if container.state != ContainerState.PENDING:
                raise ValueError(f"Invalid container state: {container.state}")
            
            # Start container based on platform
            if container.platform == ContainerPlatform.DOCKER:
                success = await self._start_docker_container(container)
            elif container.platform == ContainerPlatform.KUBERNETES:
                success = await self._start_kubernetes_container(container)
            elif container.platform == ContainerPlatform.CUSTOM:
                success = await self._start_custom_container(container)
            else:
                raise ValueError(f"Unsupported platform: {container.platform}")
            
            if success:
                container.state = ContainerState.RUNNING
                container.started_at = datetime.utcnow()
                await self._broadcast_container_event(
                    'container_started',
                    container
                )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error starting container: {str(e)}")
            return False

    async def stop_container(
        self,
        container_id: str,
        timeout: int = 30
    ) -> bool:
        """Stop container."""
        try:
            container = self.containers.get(container_id)
            if not container:
                raise ValueError(f"Container not found: {container_id}")
            
            if container.state != ContainerState.RUNNING:
                raise ValueError(f"Invalid container state: {container.state}")
            
            # Stop container based on platform
            if container.platform == ContainerPlatform.DOCKER:
                success = await self._stop_docker_container(container, timeout)
            elif container.platform == ContainerPlatform.KUBERNETES:
                success = await self._stop_kubernetes_container(container, timeout)
            elif container.platform == ContainerPlatform.CUSTOM:
                success = await self._stop_custom_container(container, timeout)
            else:
                raise ValueError(f"Unsupported platform: {container.platform}")
            
            if success:
                container.state = ContainerState.STOPPED
                container.finished_at = datetime.utcnow()
                await self._broadcast_container_event(
                    'container_stopped',
                    container
                )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error stopping container: {str(e)}")
            return False

    async def delete_container(
        self,
        container_id: str,
        force: bool = False
    ) -> bool:
        """Delete container."""
        try:
            container = self.containers.get(container_id)
            if not container:
                raise ValueError(f"Container not found: {container_id}")
            
            # Stop container if running
            if container.state == ContainerState.RUNNING:
                if not force:
                    raise ValueError("Container is running. Use force=True to delete")
                await self.stop_container(container_id)
            
            # Delete container based on platform
            if container.platform == ContainerPlatform.DOCKER:
                success = await self._delete_docker_container(container)
            elif container.platform == ContainerPlatform.KUBERNETES:
                success = await self._delete_kubernetes_container(container)
            elif container.platform == ContainerPlatform.CUSTOM:
                success = await self._delete_custom_container(container)
            else:
                raise ValueError(f"Unsupported platform: {container.platform}")
            
            if success:
                container.state = ContainerState.DELETED
                await self._broadcast_container_event(
                    'container_deleted',
                    container
                )
                del self.containers[container_id]
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error deleting container: {str(e)}")
            return False

    async def get_container_logs(
        self,
        container_id: str,
        lines: Optional[int] = None,
        since: Optional[datetime] = None
    ) -> List[str]:
        """Get container logs."""
        try:
            container = self.containers.get(container_id)
            if not container:
                raise ValueError(f"Container not found: {container_id}")
            
            # Get logs based on platform
            if container.platform == ContainerPlatform.DOCKER:
                return await self._get_docker_logs(container, lines, since)
            elif container.platform == ContainerPlatform.KUBERNETES:
                return await self._get_kubernetes_logs(container, lines, since)
            elif container.platform == ContainerPlatform.CUSTOM:
                return await self._get_custom_logs(container, lines, since)
            else:
                raise ValueError(f"Unsupported platform: {container.platform}")
            
        except Exception as e:
            self.logger.error(f"Error getting container logs: {str(e)}")
            return []

    async def execute_command(
        self,
        container_id: str,
        command: Union[str, List[str]],
        timeout: int = 30
    ) -> Tuple[int, str, str]:
        """Execute command in container."""
        try:
            container = self.containers.get(container_id)
            if not container:
                raise ValueError(f"Container not found: {container_id}")
            
            if container.state != ContainerState.RUNNING:
                raise ValueError(f"Container not running: {container_id}")
            
            # Execute command based on platform
            if container.platform == ContainerPlatform.DOCKER:
                return await self._execute_docker_command(container, command, timeout)
            elif container.platform == ContainerPlatform.KUBERNETES:
                return await self._execute_kubernetes_command(container, command, timeout)
            elif container.platform == ContainerPlatform.CUSTOM:
                return await self._execute_custom_command(container, command, timeout)
            else:
                raise ValueError(f"Unsupported platform: {container.platform}")
            
        except Exception as e:
            self.logger.error(f"Error executing command: {str(e)}")
            return (-1, "", str(e))

    async def copy_to_container(
        self,
        container_id: str,
        source_path: str,
        target_path: str
    ) -> bool:
        """Copy file/directory to container."""
        try:
            container = self.containers.get(container_id)
            if not container:
                raise ValueError(f"Container not found: {container_id}")
            
            # Copy based on platform
            if container.platform == ContainerPlatform.DOCKER:
                return await self._copy_to_docker(container, source_path, target_path)
            elif container.platform == ContainerPlatform.KUBERNETES:
                return await self._copy_to_kubernetes(container, source_path, target_path)
            elif container.platform == ContainerPlatform.CUSTOM:
                return await self._copy_to_custom(container, source_path, target_path)
            else:
                raise ValueError(f"Unsupported platform: {container.platform}")
            
        except Exception as e:
            self.logger.error(f"Error copying to container: {str(e)}")
            return False

    async def copy_from_container(
        self,
        container_id: str,
        source_path: str,
        target_path: str
    ) -> bool:
        """Copy file/directory from container."""
        try:
            container = self.containers.get(container_id)
            if not container:
                raise ValueError(f"Container not found: {container_id}")
            
            # Copy based on platform
            if container.platform == ContainerPlatform.DOCKER:
                return await self._copy_from_docker(container, source_path, target_path)
            elif container.platform == ContainerPlatform.KUBERNETES:
                return await self._copy_from_kubernetes(container, source_path, target_path)
            elif container.platform == ContainerPlatform.CUSTOM:
                return await self._copy_from_custom(container, source_path, target_path)
            else:
                raise ValueError(f"Unsupported platform: {container.platform}")
            
        except Exception as e:
            self.logger.error(f"Error copying from container: {str(e)}")
            return False

    async def get_container_metrics(
        self,
        container_id: str
    ) -> Dict:
        """Get container metrics."""
        try:
            container = self.containers.get(container_id)
            if not container:
                raise ValueError(f"Container not found: {container_id}")
            
            return container.metrics
            
        except Exception as e:
            self.logger.error(f"Error getting container metrics: {str(e)}")
            return {}

    def _prepare_container_config(
        self,
        name: str,
        image: str,
        config: Dict
    ) -> Dict:
        """Prepare container configuration."""
        # Base configuration
        container_config = {
            'name': name,
            'image': image,
            'hostname': name,
            'env': config.get('env', {}),
            'command': config.get('command'),
            'working_dir': config.get('working_dir'),
            'volumes': config.get('volumes', []),
            'ports': config.get('ports', []),
            'labels': config.get('labels', {}),
            'restart_policy': config.get('restart_policy', 'no'),
            'network_mode': config.get('network_mode', 'bridge'),
            'resources': {
                'memory': config.get('memory_limit', self.container_memory_limit),
                'cpu': config.get('cpu_limit', self.container_cpu_limit)
            }
        }
        
        # Platform-specific configurations
        if 'platform_config' in config:
            container_config.update(config['platform_config'])
        
        return container_config

    def _generate_container_id(
        self,
        name: str,
        image: str,
        platform: ContainerPlatform
    ) -> str:
        """Generate unique container ID."""
        data = f"{name}:{image}:{platform.value}:{datetime.utcnow().isoformat()}"
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    async def _broadcast_container_event(
        self,
        event_type: str,
        container: Container
    ):
        """Broadcast container event."""
        try:
            if self.redis_pool:
                event = {
                    'event_type': event_type,
                    'container_id': container.container_id,
                    'name': container.name,
                    'platform': container.platform.value,
                    'state': container.state.value,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                await self.redis_pool.publish(
                    'container_events',
                    json.dumps(event)
                )
            
        except Exception as e:
            self.logger.error(f"Error broadcasting event: {str(e)}")

    async def _container_monitor_loop(self):
        """Background task for monitoring containers."""
        while True:
            try:
                for container in list(self.containers.values()):
                    try:
                        # Update container state
                        if container.platform == ContainerPlatform.DOCKER:
                            await self._update_docker_state(container)
                        elif container.platform == ContainerPlatform.KUBERNETES:
                            await self._update_kubernetes_state(container)
                        elif container.platform == ContainerPlatform.CUSTOM:
                            await self._update_custom_state(container)
                        
                    except Exception as e:
                        self.logger.error(f"Error updating container state: {str(e)}")
                        container.state = ContainerState.UNKNOWN
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in container monitor: {str(e)}")
                await asyncio.sleep(5)

    async def _metrics_collector_loop(self):
        """Background task for collecting container metrics."""
        while True:
            try:
                for container in list(self.containers.values()):
                    if container.state == ContainerState.RUNNING:
                        try:
                            # Collect metrics based on platform
                            if container.platform == ContainerPlatform.DOCKER:
                                metrics = await self._collect_docker_metrics(container)
                            elif container.platform == ContainerPlatform.KUBERNETES:
                                metrics = await self._collect_kubernetes_metrics(container)
                            elif container.platform == ContainerPlatform.CUSTOM:
                                metrics = await self._collect_custom_metrics(container)
                            else:
                                continue
                            
                            container.metrics = metrics
                            
                        except Exception as e:
                            self.logger.error(f"Error collecting metrics: {str(e)}")
                
                await asyncio.sleep(self.metrics_interval)
                
            except Exception as e:
                self.logger.error(f"Error in metrics collector: {str(e)}")
                await asyncio.sleep(5)

    async def _cleanup_loop(self):
        """Background task for cleaning up deleted containers."""
        while True:
            try:
                cutoff_time = datetime.utcnow() - timedelta(
                    days=self.container_retention_days
                )
                
                for container_id in list(self.containers.keys()):
                    container = self.containers[container_id]
                    
                    # Clean up old deleted containers
                    if container.state == ContainerState.DELETED and \
                       container.finished_at and \
                       container.finished_at < cutoff_time:
                        del self.containers[container_id]
                
                await asyncio.sleep(self.cleanup_interval)
                
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {str(e)}")
                await asyncio.sleep(60)

    async def _initialize_clients(self):
        """Initialize platform clients."""
        try:
            # Initialize Docker client
            if 'docker' in self.config:
                self.platform_clients['docker'] = aiodocker.Docker()
            
            # Initialize Kubernetes client
            if 'kubernetes' in self.config:
                k8s.config.load_config()
                self.platform_clients['kubernetes'] = k8s.client
            
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
        'docker': {
            'base_url': 'unix://var/run/docker.sock'
        },
        'kubernetes': {
            'config_path': '~/.kube/config'
        },
        'redis_url': 'redis://localhost',
        'max_containers': 100,
        'container_memory_limit': '512m',
        'container_cpu_limit': '500m',
        'monitoring_interval': 10,
        'metrics_interval': 30,
        'cleanup_interval': 3600,
        'container_retention_days': 7
    }
    
    # Initialize container manager
    manager = ContainerManager(config)
    
    # Example container operations
    async def main():
        # Create and start container
        container = await manager.create_container(
            name='test-container',
            image='nginx:latest',
            platform=ContainerPlatform.DOCKER,
            config={
                'ports': ['80:80'],
                'env': {'NGINX_PORT': '80'},
                'restart_policy': 'unless-stopped'
            },
            metadata={'environment': 'test'}
        )
        
        print(f"Created container: {container.container_id}")
        
        # Start container
        success = await manager.start_container(container.container_id)
        if success:
            print("Container started successfully")
        
        # Wait a bit
        await asyncio.sleep(30)
        
        # Get container logs
        logs = await manager.get_container_logs(
            container.container_id,
            lines=100
        )
        print("\nContainer logs:")
        for log in logs:
            print(log)
        
        # Get metrics
        metrics = await manager.get_container_metrics(container.container_id)
        print("\nContainer metrics:")
        print(json.dumps(metrics, indent=2))
        
        # Execute command
        exit_code, stdout, stderr = await manager.execute_command(
            container.container_id,
            ["nginx", "-v"]
        )
        print(f"\nCommand output (exit code: {exit_code}):")
        print(stdout)
        
        # Stop and delete container
        await manager.stop_container(container.container_id)
        await manager.delete_container(container.container_id)
        print("\nContainer deleted")
    
    # Run example
    asyncio.run(main())