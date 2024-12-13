import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys
import os
import json

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import Settings
from app.api.models import (
    ResourceConfig, ContainerSpec, ScalingRequest,
    PatternDefinition, MetricsQuery, ValidationRule
)
from app.api.client import APIClient
from app.core.logging import setup_logging

class AdvancedScalingExample:
    """
    Demonstrates advanced scaling scenarios and patterns.
    """
    
    def __init__(self):
        """Initialize example."""
        self.client = APIClient(
            base_url="http://localhost:8000",
            api_key="your-api-key"
        )
        self.logger = logging.getLogger(__name__)

    async def predictive_scaling_pattern(self) -> str:
        """Create a predictive scaling pattern."""
        pattern = PatternDefinition(
            name="predictive-scaling",
            version="1.0.0",
            type="scaling",
            parameters={
                "min_replicas": 1,
                "max_replicas": 10,
                "prediction_window": "1h",
                "metrics": ["cpu_usage", "request_rate"],
                "target_metrics": {
                    "cpu_usage": 75,
                    "request_rate": 1000
                }
            },
            rules=[
                # Prediction-based scaling
                {
                    "condition": "predicted_cpu_usage > target_metrics.cpu_usage",
                    "action": "scale_up_predictive",
                    "parameters": {
                        "window": "prediction_window",
                        "confidence": 0.8
                    }
                },
                # Immediate response rules
                {
                    "condition": "current_cpu_usage > (target_metrics.cpu_usage * 1.2)",
                    "action": "scale_up_immediate",
                    "cooldown": "60s"
                },
                # Gradual scale-down
                {
                    "condition": "current_cpu_usage < (target_metrics.cpu_usage * 0.6)",
                    "action": "scale_down_gradual",
                    "cooldown": "300s"
                }
            ],
            actions=[
                {
                    "name": "scale_up_predictive",
                    "type": "scaling",
                    "parameters": {
                        "calculation": "predicted_replicas",
                        "max": "max_replicas",
                        "rate": "gradual"
                    }
                },
                {
                    "name": "scale_up_immediate",
                    "type": "scaling",
                    "parameters": {
                        "delta": 2,
                        "max": "max_replicas"
                    }
                },
                {
                    "name": "scale_down_gradual",
                    "type": "scaling",
                    "parameters": {
                        "delta": -1,
                        "min": "min_replicas",
                        "delay": "60s"
                    }
                }
            ]
        )
        
        pattern_id = await self.client.create_pattern(pattern)
        self.logger.info(f"Created predictive scaling pattern: {pattern_id}")
        return pattern_id

    async def create_validation_rules(self) -> List[str]:
        """Create validation rules for scaling operations."""
        rules = [
            ValidationRule(
                rule_id="cost-limit",
                rule_type="constraint",
                scope="scaling",
                conditions=[
                    {
                        "metric": "estimated_cost",
                        "operator": "less_than",
                        "value": 1000
                    }
                ],
                actions=[
                    {
                        "type": "limit",
                        "parameters": {
                            "max_replicas": 5
                        }
                    }
                ]
            ),
            ValidationRule(
                rule_id="scaling-rate",
                rule_type="rate-limit",
                scope="scaling",
                conditions=[
                    {
                        "metric": "scale_operations",
                        "window": "5m",
                        "max_count": 3
                    }
                ],
                actions=[
                    {
                        "type": "delay",
                        "parameters": {
                            "duration": "5m"
                        }
                    }
                ]
            )
        ]
        
        rule_ids = []
        for rule in rules:
            rule_id = await self.client.create_validation_rule(rule)
            rule_ids.append(rule_id)
            
        self.logger.info(f"Created validation rules: {rule_ids}")
        return rule_ids

    async def deploy_scalable_application(self) -> str:
        """Deploy application with scalable configuration."""
        config = ResourceConfig(
            cpu="1000m",
            memory="1Gi",
            disk="20Gi",
            env_vars={
                "APP_ENV": "production",
                "SCALING_ENABLED": "true"
            },
            labels={
                "app": "scalable-example",
                "scaling": "enabled"
            }
        )
        
        container = ContainerSpec(
            name="scalable-app",
            image="example/scalable-app:latest",
            ports=[{"containerPort": 8080}],
            resources=config,
            health_check={
                "httpGet": {
                    "path": "/health",
                    "port": 8080
                },
                "initialDelaySeconds": 10,
                "periodSeconds": 30
            }
        )
        
        deployment_id = await self.client.deploy_container(container)
        self.logger.info(f"Deployed scalable application: {deployment_id}")
        return deployment_id

    async def monitor_scaling_operations(
        self,
        deployment_id: str,
        duration: int = 3600
    ):
        """Monitor scaling operations for specified duration."""
        end_time = datetime.utcnow() + timedelta(seconds=duration)
        
        while datetime.utcnow() < end_time:
            # Get current metrics
            metrics = await self.client.get_metrics(
                resource_id=deployment_id,
                metrics=["cpu_usage", "memory_usage", "request_rate"],
                duration="5m"
            )
            
            # Get scaling events
            events = await self.client.get_scaling_events(
                resource_id=deployment_id,
                start_time=datetime.utcnow() - timedelta(minutes=5)
            )
            
            # Get current state
            state = await self.client.get_deployment_status(deployment_id)
            
            self.logger.info("\nCurrent Status:")
            self.logger.info(f"Replicas: {state['replicas']}")
            self.logger.info(f"CPU Usage: {metrics['cpu_usage']}%")
            self.logger.info(f"Memory Usage: {metrics['memory_usage']}Mi")
            self.logger.info(f"Request Rate: {metrics['request_rate']}/s")
            
            if events:
                self.logger.info("\nRecent Scaling Events:")
                for event in events:
                    self.logger.info(
                        f"Time: {event['timestamp']}, "
                        f"Action: {event['action']}, "
                        f"Reason: {event['reason']}"
                    )
            
            await asyncio.sleep(60)

    async def simulate_load(self, deployment_id: str):
        """Simulate varying load patterns."""
        # Create load pattern
        pattern = [
            {"cpu": 50, "requests": 500, "duration": 300},  # Normal load
            {"cpu": 85, "requests": 1200, "duration": 300},  # High load
            {"cpu": 30, "requests": 200, "duration": 300},  # Low load
            {"cpu": 95, "requests": 2000, "duration": 300},  # Spike
        ]
        
        for load in pattern:
            self.logger.info(f"\nApplying load: {load}")
            
            # Simulate load
            await self.client.simulate_load(
                resource_id=deployment_id,
                parameters=load
            )
            
            await asyncio.sleep(load["duration"])

async def main():
    """Run advanced scaling examples."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        example = AdvancedScalingExample()
        
        # Create scaling pattern
        pattern_id = await example.predictive_scaling_pattern()
        
        # Create validation rules
        rule_ids = await example.create_validation_rules()
        
        # Deploy application
        deployment_id = await example.deploy_scalable_application()
        
        # Apply pattern to deployment
        await example.client.apply_pattern(
            pattern_id=pattern_id,
            resource_id=deployment_id,
            parameters={
                "min_replicas": 2,
                "max_replicas": 8,
                "target_metrics": {
                    "cpu_usage": 70,
                    "request_rate": 800
                }
            }
        )
        
        # Start monitoring in background
        monitor_task = asyncio.create_task(
            example.monitor_scaling_operations(deployment_id)
        )
        
        # Simulate load patterns
        await example.simulate_load(deployment_id)
        
        # Wait for monitoring to complete
        await monitor_task
        
        # Cleanup
        await example.client.delete_deployment(deployment_id)
        logger.info("\nExample completed and cleaned up")
        
    except Exception as e:
        logger.error(f"Error in advanced scaling example: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())