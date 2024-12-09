# src/microservices/execution_service.py

from fastapi import FastAPI, HTTPException
from typing import Dict, List, Any
import docker
from enum import Enum
import asyncio
import logging
from datetime import datetime
import httpx

class ScalingType(Enum):
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"

class ScalingDirection(Enum):
    UP = "up"
    DOWN = "down"
    OUT = "out"
    IN = "in"

class ExecutionService:
    def __init__(self, host: str = "0.0.0.0", port: int = 8003):
        self.app = FastAPI()
        self.host = host
        self.port = port
        self.docker_client = docker.from_env()
        self.scaling_history: List[Dict[str, Any]] = []
        self.setup_routes()
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger("ExecutionService")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def setup_routes(self):
        @self.app.post("/scale/vertical")
        async def vertical_scale(scaling_request: Dict[str, Any]):
            try:
                result = await self._execute_vertical_scaling(scaling_request)
                return {"status": "success", "result": result}
            except Exception as e:
                self.logger.error(f"Vertical scaling failed: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/scale/horizontal")
        async def horizontal_scale(scaling_request: Dict[str, Any]):
            try:
                result = await self._execute_horizontal_scaling(scaling_request)
                return {"status": "success", "result": result}
            except Exception as e:
                self.logger.error(f"Horizontal scaling failed: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/history/{container_id}")
        async def get_scaling_history(container_id: str):
            history = [
                record for record in self.scaling_history 
                if record["container_id"] == container_id
            ]
            return {"history": history}

    async def _execute_vertical_scaling(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute vertical scaling operation"""
        container_id = request["container_id"]
        resources = request["resources"]
        direction = request["direction"]

        try:
            container = self.docker_client.containers.get(container_id)
            current_resources = self._get_container_resources(container)
            new_resources = self._calculate_new_resources(
                current_resources, 
                resources, 
                direction
            )

            # Apply new resource limits
            container.update(
                cpu_quota=new_resources["cpu_quota"],
                cpu_period=new_resources["cpu_period"],
                mem_limit=new_resources["memory"]
            )

            # Record scaling action
            self._record_scaling_action(
                container_id,
                ScalingType.VERTICAL,
                direction,
                current_resources,
                new_resources
            )

            self.logger.info(
                f"Vertical scaling successful for container {container_id}"
            )
            return new_resources

        except Exception as e:
            self.logger.error(
                f"Vertical scaling failed for container {container_id}: {str(e)}"
            )
            raise

    async def _execute_horizontal_scaling(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute horizontal scaling operation"""
        container_id = request["container_id"]
        replicas = request["replicas"]
        direction = request["direction"]

        try:
            container = self.docker_client.containers.get(container_id)
            current_config = container.attrs

            if direction == ScalingDirection.OUT.value:
                result = await self._scale_out(container, replicas, current_config)
            else:
                result = await self._scale_in(container_id, replicas)

            # Record scaling action
            self._record_scaling_action(
                container_id,
                ScalingType.HORIZONTAL,
                direction,
                {"replicas": len(self._get_container_replicas(container_id))},
                {"replicas": replicas}
            )

            self.logger.info(
                f"Horizontal scaling successful for container {container_id}"
            )
            return result

        except Exception as e:
            self.logger.error(
                f"Horizontal scaling failed for container {container_id}: {str(e)}"
            )
            raise

    async def _scale_out(self, container: docker.models.containers.Container, 
                        replicas: int, config: Dict[str, Any]) -> Dict[str, Any]:
        """Scale out container by creating new replicas"""
        new_containers = []
        base_name = container.name
        
        for i in range(replicas):
            try:
                new_container = self.docker_client.containers.run(
                    image=container.image,
                    name=f"{base_name}_replica_{i}",
                    detach=True,
                    environment=config.get('Config', {}).get('Env', []),
                    network_mode=config.get('HostConfig', {}).get('NetworkMode'),
                    volumes=config.get('HostConfig', {}).get('Binds', []),
                    cpu_quota=config.get('HostConfig', {}).get('CpuQuota'),
                    mem_limit=config.get('HostConfig', {}).get('Memory'),
                )
                new_containers.append(new_container.id)
            except Exception as e:
                self.logger.error(f"Failed to create replica {i}: {str(e)}")
                # Cleanup created containers on failure
                for container_id in new_containers:
                    try:
                        self.docker_client.containers.get(container_id).remove(force=True)
                    except:
                        pass
                raise

        return {"new_containers": new_containers}

    async def _scale_in(self, container_id: str, target_replicas: int) -> Dict[str, Any]:
        """Scale in by removing excess replicas"""
        replicas = self._get_container_replicas(container_id)
        removed_containers = []

        while len(replicas) > target_replicas:
            container_to_remove = replicas.pop()
            try:
                container = self.docker_client.containers.get(container_to_remove)
                container.stop()
                container.remove()
                removed_containers.append(container_to_remove)
            except Exception as e:
                self.logger.error(f"Failed to remove container {container_to_remove}: {str(e)}")

        return {"removed_containers": removed_containers}

    def _get_container_resources(self, container: docker.models.containers.Container) -> Dict[str, Any]:
        """Get current container resource allocation"""
        config = container.attrs['HostConfig']
        return {
            "cpu_quota": config.get('CpuQuota', -1),
            "cpu_period": config.get('CpuPeriod', 100000),
            "memory": config.get('Memory', 0)
        }

    def _calculate_new_resources(self, current: Dict[str, Any], 
                               requested: Dict[str, Any], 
                               direction: str) -> Dict[str, Any]:
        """Calculate new resource allocation based on scaling direction"""
        if direction == ScalingDirection.UP.value:
            return {
                "cpu_quota": int(current["cpu_quota"] * 1.5),
                "cpu_period": current["cpu_period"],
                "memory": int(current["memory"] * 1.5)
            }
        else:
            return {
                "cpu_quota": int(current["cpu_quota"] * 0.7),
                "cpu_period": current["cpu_period"],
                "memory": int(current["memory"] * 0.7)
            }

    def _get_container_replicas(self, container_id: str) -> List[str]:
        """Get list of container replicas"""
        container = self.docker_client.containers.get(container_id)
        base_name = container.name
        all_containers = self.docker_client.containers.list(all=True)
        return [
            c.id for c in all_containers 
            if c.name.startswith(base_name) and c.id != container_id
        ]

    def _record_scaling_action(self, container_id: str, 
                             scaling_type: ScalingType,
                             direction: str,
                             old_config: Dict[str, Any],
                             new_config: Dict[str, Any]) -> None:
        """Record scaling action in history"""
        self.scaling_history.append({
            "container_id": container_id,
            "timestamp": datetime.now().isoformat(),
            "scaling_type": scaling_type.value,
            "direction": direction,
            "old_config": old_config,
            "new_config": new_config
        })

    async def _validate_scaling_request(self, request: Dict[str, Any]) -> bool:
        """Validate scaling request parameters"""
        required_fields = {
            "vertical": ["container_id", "resources", "direction"],
            "horizontal": ["container_id", "replicas", "direction"]
        }
        
        scaling_type = request.get("scaling_type")
        if scaling_type not in required_fields:
            return False
            
        return all(field in request for field in required_fields[scaling_type])

    async def _notify_scaling_completion(self, container_id: str, 
                                       scaling_type: str, 
                                       result: Dict[str, Any]) -> None:
        """Notify other services about scaling completion"""
        notification = {
            "container_id": container_id,
            "scaling_type": scaling_type,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        # Implement notification logic (e.g., message queue, HTTP webhook)
        self.logger.info(f"Scaling completion notification: {notification}")

# Example usage
if __name__ == "__main__":
    import uvicorn
    
    execution_service = ExecutionService()
    uvicorn.run(execution_service.app, host="0.0.0.0", port=8003)