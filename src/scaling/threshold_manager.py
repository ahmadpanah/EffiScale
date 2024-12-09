# src/scaling/threshold_manager.py

import numpy as np
from typing import Dict, List, Any

class AdaptiveThresholdManager:
    def __init__(self, initial_threshold: float = 0.75, learning_rate: float = 0.1):
        self.base_threshold = initial_threshold
        self.learning_rate = learning_rate
        self.historical_data: List[Dict[str, Any]] = []

    def update_threshold(self, metrics: Dict[str, Any], performance_score: float) -> float:
        """
        Updates threshold based on historical data and current performance
        """
        self.historical_data.append({
            "metrics": metrics,
            "performance": performance_score
        })

        trend = self._calculate_trend()
        load = self._calculate_current_load(metrics)
        
        new_threshold = (
            self.base_threshold +
            self.learning_rate * trend +
            self.learning_rate * load
        )

        # Ensure threshold stays within reasonable bounds
        return np.clip(new_threshold, 0.1, 0.9)

    def _calculate_trend(self) -> float:
        if len(self.historical_data) < 2:
            return 0.0

        recent_metrics = [d["metrics"] for d in self.historical_data[-10:]]
        return np.mean([m["cpu_usage"] for m in recent_metrics])

    def _calculate_current_load(self, metrics: Dict[str, Any]) -> float:
        return max(metrics["cpu_usage"], metrics["memory_usage"])