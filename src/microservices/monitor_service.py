# src/microservices/monitor_service.py

from fastapi import FastAPI, HTTPException, BackgroundTasks
from typing import Dict, List, Any, Optional
import docker
import psutil
import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
from pydantic import BaseModel
import uvicorn
import json
import aiohttp
from collections import deque

class MetricData(BaseModel):
    container_id: str
    cpu_usage: float
    memory_usage: float
    network_rx: float
    network_tx: float
    disk_read: float
    disk_write: float
    timestamp: str

class MonitoringConfig(BaseModel):
    interval: int = 5
    retention_period: int = 3600
    alert_threshold_cpu: float = 0.8
    alert_threshold_memory: float = 0.8
    enable_predictions: bool = True

class MonitoringService:
    def __init__(self, host: str = "0.0.0.0", port: int = 8001):
        self.app = FastAPI(title="Container Monitoring Service")
        self.host = host
        self.port = port
        self.docker_client = docker.from_env()
        self.metric_buffer: Dict[str, deque] = {}
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.config = MonitoringConfig()
        self.logger = self._setup_logger()
        self.alert_endpoints: List[str] = []
        self.setup_routes()
        self.moving_averages: Dict[str, Dict[str, deque]] = {}
        self.prediction_models: Dict[str, Any] = {}

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("MonitoringService")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def setup_routes(self):
        @self.app.get("/")
        async def root():
            return {"status": "running", "service": "Container Monitoring"}

        @self.app.get("/metrics/{container_id}")
        async def get_container_metrics(container_id: str) -> Dict[str, Any]:
            try:
                metrics = await self._collect_container_metrics(container_id)
                return metrics
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/metrics/{container_id}/history")
        async def get_metrics_history(
            container_id: str, 
            time_window: int = 300
        ) -> Dict[str, Any]:
            try:
                return self._get_container_history(container_id, time_window)
            except Exception as e:
                self.logger.error(f"Error retrieving metric history: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/monitor/start/{container_id}")
        async def start_monitoring(
            container_id: str, 
            background_tasks: BackgroundTasks
        ):
            try:
                if container_id not in self.monitoring_tasks:
                    self.metric_buffer[container_id] = deque(maxlen=1000)
                    task = asyncio.create_task(
                        self._continuous_monitoring(container_id)
                    )
                    self.monitoring_tasks[container_id] = task
                    return {"status": "success", "message": "Monitoring started"}
                return {"status": "warning", "message": "Already monitoring"}
            except Exception as e:
                self.logger.error(f"Error starting monitoring: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/monitor/stop/{container_id}")
        async def stop_monitoring(container_id: str):
            try:
                if container_id in self.monitoring_tasks:
                    self.monitoring_tasks[container_id].cancel()
                    del self.monitoring_tasks[container_id]
                    return {"status": "success", "message": "Monitoring stopped"}
                return {"status": "warning", "message": "Not monitoring"}
            except Exception as e:
                self.logger.error(f"Error stopping monitoring: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/config")
        async def update_config(config: MonitoringConfig):
            try:
                self.config = config
                return {"status": "success", "message": "Configuration updated"}
            except Exception as e:
                self.logger.error(f"Error updating config: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/predictions/{container_id}")
        async def get_predictions(
            container_id: str, 
            prediction_window: int = 300
        ) -> Dict[str, Any]:
            try:
                return await self._predict_resource_usage(
                    container_id, 
                    prediction_window
                )
            except Exception as e:
                self.logger.error(f"Error generating predictions: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/alerts/endpoint")
        async def add_alert_endpoint(endpoint: str):
            if endpoint not in self.alert_endpoints:
                self.alert_endpoints.append(endpoint)
            return {"status": "success", "message": "Alert endpoint added"}

    async def _collect_container_metrics(self, container_id: str) -> Dict[str, Any]:
        """Collect current metrics for a container"""
        try:
            container = self.docker_client.containers.get(container_id)
            stats = container.stats(stream=False)
            
            # Calculate CPU usage
            cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
                       stats["precpu_stats"]["cpu_usage"]["total_usage"]
            system_delta = stats["cpu_stats"]["system_cpu_usage"] - \
                          stats["precpu_stats"]["system_cpu_usage"]
            cpu_usage = (cpu_delta / system_delta) * \
                       len(stats["cpu_stats"]["cpu_usage"]["percpu_usage"])

            # Calculate memory usage
            memory_usage = stats["memory_stats"]["usage"] / \
                         stats["memory_stats"]["limit"]

            # Calculate network I/O
            network_stats = stats["networks"]["eth0"]
            
            metrics = {
                "container_id": container_id,
                "cpu_usage": round(cpu_usage, 4),
                "memory_usage": round(memory_usage, 4),
                "network_rx": network_stats["rx_bytes"],
                "network_tx": network_stats["tx_bytes"],
                "disk_read": stats["blkio_stats"]["io_service_bytes_recursive"][0]["value"],
                "disk_write": stats["blkio_stats"]["io_service_bytes_recursive"][1]["value"],
                "timestamp": datetime.now().isoformat()
            }

            await self._update_moving_averages(container_id, metrics)
            await self._check_alerts(metrics)
            
            return metrics

        except Exception as e:
            self.logger.error(f"Error collecting metrics for {container_id}: {str(e)}")
            raise

    async def _continuous_monitoring(self, container_id: str):
        """Continuously monitor a container"""
        try:
            while True:
                metrics = await self._collect_container_metrics(container_id)
                self.metric_buffer[container_id].append(metrics)
                await asyncio.sleep(self.config.interval)
        except asyncio.CancelledError:
            self.logger.info(f"Monitoring stopped for container {container_id}")
        except Exception as e:
            self.logger.error(
                f"Error in continuous monitoring for {container_id}: {str(e)}"
            )
            raise

    def _get_container_history(
        self, 
        container_id: str, 
        time_window: int
    ) -> Dict[str, Any]:
        """Get historical metrics for a container"""
        if container_id not in self.metric_buffer:
            raise HTTPException(
                status_code=404, 
                detail="No metrics found for container"
            )

        cutoff_time = datetime.now() - timedelta(seconds=time_window)
        history = [
            metric for metric in self.metric_buffer[container_id]
            if datetime.fromisoformat(metric["timestamp"]) > cutoff_time
        ]

        return {
            "container_id": container_id,
            "time_window": time_window,
            "metrics": history,
            "statistics": self._calculate_statistics(history)
        }

    async def _update_moving_averages(
        self, 
        container_id: str, 
        metrics: Dict[str, Any]
    ):
        """Update moving averages for metrics"""
        if container_id not in self.moving_averages:
            self.moving_averages[container_id] = {
                "cpu": deque(maxlen=10),
                "memory": deque(maxlen=10),
                "network_rx": deque(maxlen=10),
                "network_tx": deque(maxlen=10)
            }

        self.moving_averages[container_id]["cpu"].append(metrics["cpu_usage"])
        self.moving_averages[container_id]["memory"].append(metrics["memory_usage"])
        self.moving_averages[container_id]["network_rx"].append(metrics["network_rx"])
        self.moving_averages[container_id]["network_tx"].append(metrics["network_tx"])

    async def _check_alerts(self, metrics: Dict[str, Any]):
        """Check if metrics exceed thresholds and send alerts"""
        alerts = []

        if metrics["cpu_usage"] > self.config.alert_threshold_cpu:
            alerts.append({
                "type": "cpu_high",
                "value": metrics["cpu_usage"],
                "threshold": self.config.alert_threshold_cpu
            })

        if metrics["memory_usage"] > self.config.alert_threshold_memory:
            alerts.append({
                "type": "memory_high",
                "value": metrics["memory_usage"],
                "threshold": self.config.alert_threshold_memory
            })

        if alerts:
            await self._send_alerts(metrics["container_id"], alerts)

    async def _send_alerts(self, container_id: str, alerts: List[Dict[str, Any]]):
        """Send alerts to registered endpoints"""
        alert_data = {
            "container_id": container_id,
            "timestamp": datetime.now().isoformat(),
            "alerts": alerts
        }

        async with aiohttp.ClientSession() as session:
            for endpoint in self.alert_endpoints:
                try:
                    async with session.post(
                        endpoint, 
                        json=alert_data
                    ) as response:
                        if response.status != 200:
                            self.logger.error(
                                f"Failed to send alert to {endpoint}: {response.status}"
                            )
                except Exception as e:
                    self.logger.error(
                        f"Error sending alert to {endpoint}: {str(e)}"
                    )

    async def _predict_resource_usage(
        self, 
        container_id: str, 
        prediction_window: int
    ) -> Dict[str, Any]:
        """Predict future resource usage"""
        if not self.config.enable_predictions:
            return {"status": "predictions_disabled"}

        if container_id not in self.metric_buffer:
            raise HTTPException(
                status_code=404, 
                detail="No metrics found for container"
            )

        history = list(self.metric_buffer[container_id])
        if len(history) < 10:
            return {"status": "insufficient_data"}

        predictions = {
            "cpu_usage": self._linear_prediction(
                [m["cpu_usage"] for m in history], 
                prediction_window
            ),
            "memory_usage": self._linear_prediction(
                [m["memory_usage"] for m in history], 
                prediction_window
            )
        }

        return {
            "container_id": container_id,
            "prediction_window": prediction_window,
            "predictions": predictions
        }

    def _linear_prediction(
        self, 
        values: List[float], 
        prediction_window: int
    ) -> List[float]:
        """Simple linear prediction of future values"""
        x = np.arange(len(values))
        coefficients = np.polyfit(x, values, 1)
        model = np.poly1d(coefficients)
        
        future_x = np.arange(
            len(values), 
            len(values) + prediction_window
        )
        predictions = model(future_x)
        
        return predictions.tolist()

    def _calculate_statistics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistical measures for metrics"""
        if not metrics:
            return {}

        cpu_values = [m["cpu_usage"] for m in metrics]
        memory_values = [m["memory_usage"] for m in metrics]

        return {
            "cpu": {
                "mean": np.mean(cpu_values),
                "max": np.max(cpu_values),
                "min": np.min(cpu_values),
                "std": np.std(cpu_values)
            },
            "memory": {
                "mean": np.mean(memory_values),
                "max": np.max(memory_values),
                "min": np.min(memory_values),
                "std": np.std(memory_values)
            }
        }

    def run(self):
        """Run the monitoring service"""
        uvicorn.run(
            self.app, 
            host=self.host, 
            port=self.port
        )

# Example usage
if __name__ == "__main__":
    monitoring_service = MonitoringService()
    monitoring_service.run()