# src/scaling/hybrid_scaler.py

from typing import Dict, Any
from enum import Enum

class ScalingType(Enum):
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"

class HybridScaler:
    def __init__(self):
        self.vertical_thresholds = {
            "cpu": {"high": 0.8, "low": 0.2},
            "memory": {"high": 0.8, "low": 0.2}
        }
        self.horizontal_thresholds = {
            "cpu": {"high": 0.9, "low": 0.1},
            "memory": {"high": 0.9, "low": 0.1}
        }

    def determine_scaling_action(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        cpu_usage = metrics["cpu_usage"]
        memory_usage = metrics["memory_usage"]

        # Check for vertical scaling first
        if self._needs_vertical_scaling(cpu_usage, memory_usage):
            return self._create_vertical_scaling_plan(cpu_usage, memory_usage)

        # Check for horizontal scaling
        if self._needs_horizontal_scaling(cpu_usage, memory_usage):
            return self._create_horizontal_scaling_plan(cpu_usage, memory_usage)

        return {"action": "none"}

    def _needs_vertical_scaling(self, cpu_usage: float, memory_usage: float) -> bool:
        return (cpu_usage > self.vertical_thresholds["cpu"]["high"] or
                memory_usage > self.vertical_thresholds["memory"]["high"])

    def _needs_horizontal_scaling(self, cpu_usage: float, memory_usage: float) -> bool:
        return (cpu_usage > self.horizontal_thresholds["cpu"]["high"] or
                memory_usage > self.horizontal_thresholds["memory"]["high"])

    def _create_vertical_scaling_plan(self, cpu_usage: float, memory_usage: float) -> Dict[str, Any]:
        return {
            "action": "scale",
            "type": ScalingType.VERTICAL.value,
            "resources": self._calculate_new_resources(cpu_usage, memory_usage)
        }

    def _create_horizontal_scaling_plan(self, cpu_usage: float, memory_usage: float) -> Dict[str, Any]:
        return {
            "action": "scale",
            "type": ScalingType.HORIZONTAL.value,
            "instances": self._calculate_new_instances(cpu_usage, memory_usage)
        }