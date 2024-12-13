import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import Settings
from app.api.models import (
    ResourceConfig, ContainerSpec, ScalingRequest,
    PatternDefinition, MetricsQuery
)
from app.api.client import APIClient
from app.core.logging import setup_logging

async def basic_resource_management():
    """Demonstrate basic resource management operations."""
    # Initialize client
    client = APIClient(
        base_url="http://localhost:8000",
        api_key="your-api-key"
    )
    
    # Create resource configuration
    config = ResourceConfig(
        cpu="500m",
        memory="512Mi",
        disk="10Gi",
        env_vars={
            "DATABASE_URL": "postgresql://localhost:5432/app",
            "REDIS_URL": "redis://localhost:6379/0"
        },
        labels={
            "environment": "development",
            "app": "example"
        }
    )
    
    # Create container specification
    container = ContainerSpec(
        name="example-app",
        image="example/app:latest",
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
    
    # Deploy container
    deployment_id = await client.deploy_container(container)
    print(f"Deployed container with ID: {deployment_id}")
    
    # Wait for deployment
    while True:
        status = await client.get_deployment_status(deployment_id)
        if status["state"] == "running":
            break
        print(f"Waiting for deployment... Current state: {status['state']}")
        await asyncio.sleep(5)
    
    # Get metrics
    metrics = await client.get_metrics(
        resource_id=deployment_id,
        metrics=["cpu_usage", "memory_usage"],
        duration="5m"
    )
    print("\nResource metrics:")
    print(f"CPU Usage: {metrics['cpu_usage']}%")
    print(f"Memory Usage: {metrics['memory_usage']}Mi")
    
    # Cleanup
    await client.delete_deployment(deployment_id)
    print("\nDeployment cleaned up")

async def basic_pattern_usage():
    """Demonstrate basic pattern operations."""
    client = APIClient(
        base_url="http://localhost:8000",
        api_key="your-api-key"
    )
    
    # Define a basic scaling pattern
    pattern = PatternDefinition(
        name="basic-scaling",
        version="1.0.0",
        type="scaling",
        parameters={
            "min_replicas": 1,
            "max_replicas": 5,
            "target_cpu_utilization": 75
        },
        rules=[
            {
                "condition": "cpu_usage > target_cpu_utilization",
                "action": "scale_up",
                "cooldown": "300s"
            },
            {
                "condition": "cpu_usage < (target_cpu_utilization * 0.7)",
                "action": "scale_down",
                "cooldown": "300s"
            }
        ],
        actions=[
            {
                "name": "scale_up",
                "type": "scaling",
                "parameters": {
                    "delta": 1,
                    "max": "max_replicas"
                }
            },
            {
                "name": "scale_down",
                "type": "scaling",
                "parameters": {
                    "delta": -1,
                    "min": "min_replicas"
                }
            }
        ]
    )
    
    # Create pattern
    pattern_id = await client.create_pattern(pattern)
    print(f"\nCreated pattern with ID: {pattern_id}")
    
    # Apply pattern to deployment
    await client.apply_pattern(
        pattern_id=pattern_id,
        resource_id="your-deployment-id",
        parameters={
            "min_replicas": 2,
            "max_replicas": 10,
            "target_cpu_utilization": 80
        }
    )
    print("\nPattern applied to deployment")
    
    # Get pattern status
    status = await client.get_pattern_status(pattern_id)
    print(f"\nPattern status: {status}")

async def main():
    """Run basic usage examples."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Run resource management example
        logger.info("Running resource management example...")
        await basic_resource_management()
        
        # Run pattern usage example
        logger.info("\nRunning pattern usage example...")
        await basic_pattern_usage()
        
    except Exception as e:
        logger.error(f"Error in examples: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())