# main.py

import asyncio
from src.microservices.monitor_service import MonitoringService
from src.microservices.controller_service import ControllerService
from src.microservices.execution_service import ExecutionService
from src.controllers.decision_maker import DecisionMaker
import uvicorn
from fastapi import FastAPI
import logging
from typing import Dict, Any
import multiprocessing
import threading

class EffiScaleOrchestrator:
    def __init__(self):
        self.app = FastAPI(title="EffiScale Orchestrator")
        self.logger = self._setup_logger()
        
        # Initialize services with different ports
        self.monitor_service = MonitoringService(host="0.0.0.0", port=8001)
        self.controller_service = ControllerService(
            controller_id="main_controller",
            zone="main_zone",
            host="0.0.0.0",
            port=8002
        )
        self.execution_service = ExecutionService(host="0.0.0.0", port=8003)
        self.decision_maker = DecisionMaker()

        # Store service processes
        self.service_processes = {}

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("EffiScale")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def start_services(self):
        """Start all microservices in separate processes"""
        try:
            # Start Monitor Service
            monitor_process = multiprocessing.Process(
                target=self._run_monitor_service
            )
            monitor_process.start()
            self.service_processes['monitor'] = monitor_process

            # Start Controller Service
            controller_process = multiprocessing.Process(
                target=self._run_controller_service
            )
            controller_process.start()
            self.service_processes['controller'] = controller_process

            # Start Execution Service
            execution_process = multiprocessing.Process(
                target=self._run_execution_service
            )
            execution_process.start()
            self.service_processes['execution'] = execution_process

            self.logger.info("All services started successfully")

        except Exception as e:
            self.logger.error(f"Error starting services: {str(e)}")
            self.stop_services()
            raise

    def _run_monitor_service(self):
        """Run the monitor service"""
        self.logger.info("Starting Monitor Service...")
        uvicorn.run(
            self.monitor_service.app,
            host=self.monitor_service.host,
            port=self.monitor_service.port
        )

    def _run_controller_service(self):
        """Run the controller service"""
        self.logger.info("Starting Controller Service...")
        uvicorn.run(
            self.controller_service.app,
            host=self.controller_service.host,
            port=self.controller_service.port
        )

    def _run_execution_service(self):
        """Run the execution service"""
        self.logger.info("Starting Execution Service...")
        uvicorn.run(
            self.execution_service.app,
            host=self.execution_service.host,
            port=self.execution_service.port
        )

    def stop_services(self):
        """Stop all running services"""
        for service_name, process in self.service_processes.items():
            try:
                process.terminate()
                process.join()
                self.logger.info(f"Stopped {service_name} service")
            except Exception as e:
                self.logger.error(f"Error stopping {service_name} service: {str(e)}")

    async def orchestrate_scaling(self, container_id: str):
        """Orchestrate the scaling process for a container"""
        try:
            # 1. Get metrics from Monitor Service
            metrics = await self._get_metrics(container_id)

            # 2. Make scaling decision
            decision = await self.decision_maker.make_scaling_decision(
                container_id,
                metrics,
                []  # You can add historical data here
            )

            # 3. Send decision to Controller Service
            controller_response = await self._send_to_controller(
                container_id, 
                decision
            )

            # 4. Execute scaling action
            if controller_response.get('approved', False):
                execution_response = await self._execute_scaling(
                    container_id, 
                    decision
                )
                return execution_response
            
            return {"status": "no_action_needed"}

        except Exception as e:
            self.logger.error(f"Error in scaling orchestration: {str(e)}")
            raise

    async def _get_metrics(self, container_id: str) -> Dict[str, Any]:
        """Get metrics from Monitor Service"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://localhost:8001/metrics/{container_id}"
            ) as response:
                return await response.json()

    async def _send_to_controller(
        self, 
        container_id: str, 
        decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send scaling decision to Controller Service"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://localhost:8002/decision",
                json={"container_id": container_id, "decision": decision}
            ) as response:
                return await response.json()

    async def _execute_scaling(
        self, 
        container_id: str, 
        decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute scaling action via Execution Service"""
        async with aiohttp.ClientSession() as session:
            if decision['decision'] == 'vertical_scale_up':
                endpoint = f"http://localhost:8003/scale/vertical"
            else:
                endpoint = f"http://localhost:8003/scale/horizontal"

            async with session.post(
                endpoint,
                json={
                    "container_id": container_id,
                    "scaling_params": decision['parameters']
                }
            ) as response:
                return await response.json()

def main():
    # Create orchestrator instance
    orchestrator = EffiScaleOrchestrator()

    try:
        # Start all services
        orchestrator.start_services()

        # Keep the main process running
        while True:
            try:
                # You can add periodic tasks or health checks here
                asyncio.sleep(60)
            except KeyboardInterrupt:
                break

    except Exception as e:
        print(f"Error in main process: {str(e)}")
    finally:
        # Stop all services
        orchestrator.stop_services()

if __name__ == "__main__":
    main()