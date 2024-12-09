# src/microservices/storage_service.py

from fastapi import FastAPI, HTTPException
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime, timedelta

class MetricStorageService:
    def __init__(self):
        self.app = FastAPI()
        self.metrics_db = pd.DataFrame(columns=[
            'timestamp', 'container_id', 'cpu_usage',
            'memory_usage', 'scaling_action', 'performance_score'
        ])
        self.setup_routes()

    def setup_routes(self):
        @self.app.post("/metrics")
        async def store_metrics(metrics: Dict[str, Any]):
            self._store_metrics(metrics)
            return {"status": "success"}

        @self.app.get("/analysis/{container_id}")
        async def get_analysis(container_id: str, 
                             time_window: int = 3600) -> Dict[str, Any]:
            return self._analyze_metrics(container_id, time_window)

    def _store_metrics(self, metrics: Dict[str, Any]) -> None:
        new_row = pd.DataFrame([metrics])
        self.metrics_db = pd.concat([self.metrics_db, new_row], ignore_index=True)

    def _analyze_metrics(self, container_id: str, 
                        time_window: int) -> Dict[str, Any]:
        end_time = datetime.now()
        start_time = end_time - timedelta(seconds=time_window)

        container_metrics = self.metrics_db[
            (self.metrics_db['container_id'] == container_id) &
            (self.metrics_db['timestamp'] >= start_time)
        ]

        return {
            "average_cpu": container_metrics['cpu_usage'].mean(),
            "average_memory": container_metrics['memory_usage'].mean(),
            "scaling_history": container_metrics['scaling_action'].tolist(),
            "performance_trend": self._calculate_performance_trend(container_metrics)
        }

    def _calculate_performance_trend(self, metrics: pd.DataFrame) -> float:
        return metrics['performance_score'].ewm(span=10).mean().iloc[-1]