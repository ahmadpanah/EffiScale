# src/controllers/controller_federation.py

from typing import Dict, List, Any
import asyncio
import httpx
from datetime import datetime

class ControllerFederation:
    def __init__(self, controller_id: str, zone: str):
        self.controller_id = controller_id
        self.zone = zone
        self.neighbors: Dict[str, str] = {}  # controller_id: endpoint
        self.last_heartbeat: Dict[str, datetime] = {}
        self.leader = None
        self.weight = 1.0

    async def register_neighbor(self, neighbor_id: str, endpoint: str) -> None:
        self.neighbors[neighbor_id] = endpoint
        self.last_heartbeat[neighbor_id] = datetime.now()

    async def make_global_decision(self, local_decision: Dict[str, Any]) -> Dict[str, Any]:
        all_decisions = await self._collect_neighbor_decisions()
        all_decisions[self.controller_id] = local_decision
        
        return self._weighted_consensus(all_decisions)

    async def _collect_neighbor_decisions(self) -> Dict[str, Any]:
        decisions = {}
        async with httpx.AsyncClient() as client:
            for neighbor_id, endpoint in self.neighbors.items():
                try:
                    response = await client.get(f"{endpoint}/decision")
                    if response.status_code == 200:
                        decisions[neighbor_id] = response.json()
                except Exception as e:
                    print(f"Failed to get decision from {neighbor_id}: {str(e)}")
                    await self._handle_neighbor_failure(neighbor_id)
        return decisions

    def _weighted_consensus(self, decisions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        action_weights = {
            "scale_out": 0,
            "scale_in": 0,
            "scale_up": 0,
            "scale_down": 0,
            "no_action": 0
        }

        for controller_id, decision in decisions.items():
            action = decision.get("action", "no_action")
            weight = self.weight if controller_id == self.controller_id else 1.0
            action_weights[action] += weight

        best_action = max(action_weights.items(), key=lambda x: x[1])[0]
        return {"action": best_action}

    async def _handle_neighbor_failure(self, neighbor_id: str) -> None:
        del self.neighbors[neighbor_id]
        del self.last_heartbeat[neighbor_id]
        await self._redistribute_load()

    async def _redistribute_load(self) -> None:
        total_neighbors = len(self.neighbors) + 1
        self.weight = 1.0 + (1.0 / total_neighbors)