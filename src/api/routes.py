from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path, Body
from fastapi.security import OAuth2PasswordBearer
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio
import json
from enum import Enum
from pydantic import BaseModel, Field, validator
from uuid import UUID, uuid4

# Import managers and validators
from ..managers.scaling_executor import ScalingExecutor, ScalingAction, ResourceType, ScalingStatus
from ..managers.pattern_library import PatternLibrary, PatternType, Pattern
from ..managers.container_manager import ContainerManager, ContainerState, ContainerPlatform
from ..managers.rollback_manager import RollbackManager, RollbackType, RollbackStatus
from ..validators.knowledge_validator import KnowledgeValidator, ValidationScope

# API routers
scaling_router = APIRouter(prefix="/api/v1/scaling", tags=["scaling"])
pattern_router = APIRouter(prefix="/api/v1/patterns", tags=["patterns"])
container_router = APIRouter(prefix="/api/v1/containers", tags=["containers"])
rollback_router = APIRouter(prefix="/api/v1/rollbacks", tags=["rollbacks"])
monitoring_router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])

# Auth scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Request/Response Models
class ScalingRequest(BaseModel):
    action: ScalingAction
    resource_type: ResourceType
    resource_id: str
    target_state: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class PatternRequest(BaseModel):
    pattern_type: PatternType
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None

class ContainerRequest(BaseModel):
    name: str
    image: str
    platform: ContainerPlatform
    config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class RollbackRequest(BaseModel):
    rollback_type: RollbackType
    steps: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None
    triggered_by: Optional[str] = None

# Dependency Injection
async def get_scaling_executor() -> ScalingExecutor:
    # Initialize with configuration
    config = {
        "max_concurrent_operations": 5,
        "operation_timeout": 300,
        "monitoring_interval": 10
    }
    return ScalingExecutor(config)

async def get_pattern_library() -> PatternLibrary:
    config = {
        "postgres_url": "postgresql://user:password@localhost/dbname",
        "sqlite_path": "patterns.db"
    }
    return PatternLibrary(config)

async def get_container_manager() -> ContainerManager:
    config = {
        "docker": {"base_url": "unix://var/run/docker.sock"},
        "kubernetes": {"config_path": "~/.kube/config"}
    }
    return ContainerManager(config)

async def get_rollback_manager() -> RollbackManager:
    config = {
        "sqlite_path": "rollback.db",
        "redis_url": "redis://localhost"
    }
    return RollbackManager(config)

async def get_knowledge_validator() -> KnowledgeValidator:
    config = {
        "validation_level": "NORMAL",
        "max_history_size": 1000
    }
    return KnowledgeValidator(config)

# Scaling Routes
@scaling_router.post("/operations", response_model=Dict)
async def create_scaling_operation(
    request: ScalingRequest,
    background_tasks: BackgroundTasks,
    executor: ScalingExecutor = Depends(get_scaling_executor)
):
    """Create new scaling operation."""
    try:
        operation = await executor.execute_scaling(
            action=request.action,
            resource_type=request.resource_type,
            resource_id=request.resource_id,
            target_state=request.target_state,
            metadata=request.metadata
        )
        
        return {
            "operation_id": operation.operation_id,
            "status": operation.status.value
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@scaling_router.get("/operations/{operation_id}", response_model=Dict)
async def get_scaling_operation(
    operation_id: str = Path(..., title="Operation ID"),
    executor: ScalingExecutor = Depends(get_scaling_executor)
):
    """Get scaling operation details."""
    try:
        operation = executor.operations.get(operation_id)
        if not operation:
            raise HTTPException(status_code=404, detail="Operation not found")
            
        return {
            "operation_id": operation.operation_id,
            "status": operation.status.value,
            "metrics": operation.metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Pattern Routes
@pattern_router.post("/patterns", response_model=Dict)
async def create_pattern(
    request: PatternRequest,
    background_tasks: BackgroundTasks,
    library: PatternLibrary = Depends(get_pattern_library),
    validator: KnowledgeValidator = Depends(get_knowledge_validator)
):
    """Create new pattern."""
    try:
        # Validate pattern
        validation = await validator.validate_knowledge(
            pattern=request.data,
            scope=ValidationScope.PATTERN
        )
        
        if not validation.is_valid:
            raise HTTPException(
                status_code=400,
                detail={"errors": validation.errors}
            )
        
        # Add pattern
        pattern = await library.add_pattern(
            pattern_type=request.pattern_type,
            data=request.data,
            metadata=request.metadata,
            tags=set(request.tags) if request.tags else None
        )
        
        return {
            "pattern_id": pattern.pattern_id,
            "confidence": pattern.confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@pattern_router.get("/patterns", response_model=List[Dict])
async def get_patterns(
    pattern_type: Optional[PatternType] = Query(None),
    tags: Optional[List[str]] = Query(None),
    library: PatternLibrary = Depends(get_pattern_library)
):
    """Get patterns matching criteria."""
    try:
        patterns = await library.get_patterns(
            pattern_type=pattern_type,
            tags=set(tags) if tags else None
        )
        
        return [
            {
                "pattern_id": p.pattern_id,
                "pattern_type": p.pattern_type.value,
                "data": p.data,
                "confidence": p.confidence
            }
            for p in patterns
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Container Routes
@container_router.post("/containers", response_model=Dict)
async def create_container(
    request: ContainerRequest,
    background_tasks: BackgroundTasks,
    manager: ContainerManager = Depends(get_container_manager)
):
    """Create new container."""
    try:
        container = await manager.create_container(
            name=request.name,
            image=request.image,
            platform=request.platform,
            config=request.config,
            metadata=request.metadata
        )
        
        return {
            "container_id": container.container_id,
            "state": container.state.value
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@container_router.post("/containers/{container_id}/start")
async def start_container(
    container_id: str = Path(..., title="Container ID"),
    manager: ContainerManager = Depends(get_container_manager)
):
    """Start container."""
    try:
        success = await manager.start_container(container_id)
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to start container"
            )
        return {"status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@container_router.post("/containers/{container_id}/stop")
async def stop_container(
    container_id: str = Path(..., title="Container ID"),
    timeout: int = Query(30, ge=0),
    manager: ContainerManager = Depends(get_container_manager)
):
    """Stop container."""
    try:
        success = await manager.stop_container(container_id, timeout)
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to stop container"
            )
        return {"status": "stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Rollback Routes
@rollback_router.post("/rollbacks", response_model=Dict)
async def create_rollback(
    request: RollbackRequest,
    background_tasks: BackgroundTasks,
    manager: RollbackManager = Depends(get_rollback_manager)
):
    """Create new rollback operation."""
    try:
        operation = await manager.create_rollback(
            rollback_type=request.rollback_type,
            steps=request.steps,
            metadata=request.metadata,
            triggered_by=request.triggered_by
        )
        
        return {
            "operation_id": operation.operation_id,
            "status": operation.status.value
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@rollback_router.post("/rollbacks/{operation_id}/execute")
async def execute_rollback(
    operation_id: str = Path(..., title="Operation ID"),
    manager: RollbackManager = Depends(get_rollback_manager)
):
    """Execute rollback operation."""
    try:
        success = await manager.execute_rollback(operation_id)
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Rollback execution failed"
            )
        return {"status": "executed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Monitoring Routes
@monitoring_router.get("/metrics/{resource_type}/{resource_id}")
async def get_resource_metrics(
    resource_type: str = Path(..., title="Resource Type"),
    resource_id: str = Path(..., title="Resource ID"),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None)
):
    """Get resource metrics."""
    try:
        # Implementation depends on monitoring system
        metrics = {}  # Get metrics from monitoring system
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@monitoring_router.get("/health")
async def get_health_status():
    """Get system health status."""
    try:
        # Check various system components
        status = {
            "scaling": True,
            "patterns": True,
            "containers": True,
            "rollbacks": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Error Handlers
@scaling_router.exception_handler(Exception)
async def scaling_exception_handler(request, exc):
    return {"error": str(exc)}

@pattern_router.exception_handler(Exception)
async def pattern_exception_handler(request, exc):
    return {"error": str(exc)}

@container_router.exception_handler(Exception)
async def container_exception_handler(request, exc):
    return {"error": str(exc)}

@rollback_router.exception_handler(Exception)
async def rollback_exception_handler(request, exc):
    return {"error": str(exc)}