# src/microservices/controller_service.py

from fastapi import FastAPI, HTTPException
from typing import Dict, List, Any
import asyncio
import httpx
from datetime import datetime
import uvicorn
from enum import Enum

class ScalingAction(Enum):
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"

class ControllerService:
    def __init__(self, controller_id: str, zone: str, host: str = "0.0.0.0", port: int = 8000):
        self.app = FastAPI()
        self.controller_id = controller_id
        self.zone = zone
        self.host = host
        self.port = port
        self.neighbors: Dict[str, str] = {}  # controller_id: endpoint
        self.is_leader = False
        self.weight = 1.0
        self.last_heartbeat: Dict[str, datetime] = {}
        self.setup_routes()

    def setup_routes(self):
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "controller_id": self.controller_id, "zone": self.zone}

        @self.app.post("/register")
        async def register_neighbor(neighbor_info: Dict[str, str]):
            await self.register_neighbor(neighbor_info["controller_id"], neighbor_info["endpoint"])
            return {"status": "registered"}

        @self.app.get("/neighbors")
        async def get_neighbors():
            return {"neighbors": self.neighbors}

        @self.app.post("/decision")
        async def make_scaling_decision(metrics: Dict[str, Any]):
            try:
                local_decision = await self._make_local_decision(metrics)
                global_decision = await self._get_consensus_decision(local_decision)
                return global_decision
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/leader")
        async def get_leader_status():
            return {"is_leader": self.is_leader}

        @self.app.post("/metrics/threshold")
        async def update_thresholds(metrics: Dict[str, Any]):
            return await self._update_adaptive_thresholds(metrics)

    async def register_neighbor(self, neighbor_id: str, endpoint: str) -> None:
        """Register a new neighboring controller"""
        self.neighbors[neighbor_id] = endpoint
        self.last_heartbeat[neighbor_id] = datetime.now()
        await self._notify_neighbors_of_new_peer(neighbor_id, endpoint)

    async def _make_local_decision(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Make a local scaling decision based on metrics"""
        cpu_usage = metrics.get("cpu_usage", 0)
        memory_usage = metrics.get("memory_usage", 0)
        
        decision = {
            "controller_id": self.controller_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "action": self._determine_scaling_action(cpu_usage, memory_usage)
        }
        
        return decision

    async def _get_consensus_decision(self, local_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Get consensus decision from all controllers"""
        all_decisions = await self._collect_neighbor_decisions(local_decision)
        return self._weighted_consensus(all_decisions)

    async def _collect_neighbor_decisions(self, local_decision: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Collect decisions from all neighboring controllers"""
        decisions = {self.controller_id: local_decision}
        
        async with httpx.AsyncClient() as client:
            for neighbor_id, endpoint in self.neighbors.items():
                try:
                    response = await client.post(
                        f"{endpoint}/decision",
                        json=local_decision["metrics"],
                        timeout=5.0
                    )
                    if response.status_code == 200:
                        decisions[neighbor_id] = response.json()
                except Exception as e:
                    print(f"Failed to get decision from {neighbor_id}: {str(e)}")
                    await self._handle_neighbor_failure(neighbor_id)
        
        return decisions

    def _weighted_consensus(self, decisions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate weighted consensus from all decisions"""
        action_weights = {action.value: 0 for action in ScalingAction}
        
        for controller_id, decision in decisions.items():
            action = decision.get("action", ScalingAction.NO_ACTION.value)
            weight = self.weight if controller_id == self.controller_id else 1.0
            action_weights[action] += weight

        # Get action with highest weight
        consensus_action = max(action_weights.items(), key=lambda x: x[1])[0]
        
        return {
            "action": consensus_action,
            "timestamp": datetime.now().isoformat(),
            "participating_controllers": list(decisions.keys())
        }

    def _determine_scaling_action(self, cpu_usage: float, memory_usage: float) -> str:
        """Determine scaling action based on resource usage"""
        if cpu_usage > 0.9 or memory_usage > 0.9:
            return ScalingAction.SCALE_OUT.value
        elif cpu_usage > 0.75 or memory_usage > 0.75:
            return ScalingAction.SCALE_UP.value
        elif cpu_usage < 0.3 and memory_usage < 0.3:
            return ScalingAction.SCALE_IN.value
        elif cpu_usage < 0.4 and memory_usage < 0.4:
            return ScalingAction.SCALE_DOWN.value
        return ScalingAction.NO_ACTION.value

    async def _handle_neighbor_failure(self, neighbor_id: str) -> None:
        """Handle failure of a neighboring controller"""
        del self.neighbors[neighbor_id]
        del self.last_heartbeat[neighbor_id]
        await self._redistribute_load()
        await self._initiate_leader_election()

    async def _redistribute_load(self) -> None:
        """Redistribute load after neighbor failure"""
        total_neighbors = len(self.neighbors) + 1
        self.weight = 1.0 + (1.0 / total_neighbors)

    async def _initiate_leader_election(self) -> None:
        """Initiate leader election process"""
        if not self.neighbors:
            self.is_leader = True
            return

        # Simple leader election based on controller ID
        potential_leaders = [self.controller_id] + list(self.neighbors.keys())
        new_leader = min(potential_leaders)
        self.is_leader = (new_leader == self.controller_id)

    async def _update_adaptive_thresholds(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Update adaptive thresholds based on metrics"""
        # Implementation of adaptive threshold adjustment
        pass

    async def _notify_neighbors_of_new_peer(self, new_id: str, new_endpoint: str) -> None:
        """Notify all neighbors about a new peer"""
        async with httpx.AsyncClient() as client:
            for endpoint in self.neighbors.values():
                try:
                    await client.post(
                        f"{endpoint}/register",
                        json={"controller_id": new_id, "endpoint": new_endpoint},
                        timeout=5.0
                    )
                except Exception as e:
                    print(f"Failed to notify neighbor: {str(e)}")

    async def start_heartbeat_monitor(self):
        """Start monitoring heartbeats from neighbors"""
        while True:
            await self._check_neighbor_heartbeats()
            await asyncio.sleep(30)  # Check every 30 seconds

    async def _check_neighbor_heartbeats(self):
        """Check heartbeats from all neighbors"""
        current_time = datetime.now()
        failed_neighbors = []

        for neighbor_id, last_heartbeat in self.last_heartbeat.items():
            if (current_time - last_heartbeat).seconds > 60:  # 1 minute timeout
                failed_neighbors.append(neighbor_id)

        for neighbor_id in failed_neighbors:
            await self._handle_neighbor_failure(neighbor_id)

    def run(self):
        """Run the controller service"""
        asyncio.create_task(self.start_heartbeat_monitor())
        uvicorn.run(self.app, host=self.host, port=self.port)

# Example usage
if __name__ == "__main__":
    controller = ControllerService(
        controller_id="controller1",
        zone="zone1",
        host="0.0.0.0",
        port=8000
    )
    controller.run()