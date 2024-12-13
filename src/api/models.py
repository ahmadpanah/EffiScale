from typing import Dict, List, Optional, Union, Any, Set
from datetime import datetime
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum
import re
from uuid import UUID, uuid4
import json

class ResourceScope(Enum):
    """Resource access scopes."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"

class EnvironmentType(Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class ActionType(Enum):
    """Action types."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    VALIDATE = "validate"

# Base Models
class BaseRequest(BaseModel):
    """Base request model."""
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: Optional[str] = None
    environment: EnvironmentType = Field(default=EnvironmentType.DEVELOPMENT)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }

class BaseResponse(BaseModel):
    """Base response model."""
    request_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: str
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }

# Authentication Models
class TokenRequest(BaseModel):
    """Token request model."""
    username: str
    password: str
    scope: Optional[List[ResourceScope]] = Field(default_factory=list)

class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str
    expires_in: int
    scope: List[ResourceScope]
    user_id: str

class UserModel(BaseModel):
    """User model."""
    user_id: str
    username: str
    email: str
    roles: List[str]
    scopes: List[ResourceScope]
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @validator('email')
    def validate_email(cls, v):
        if not re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", v):
            raise ValueError('Invalid email format')
        return v

# Resource Models
class ResourceConfig(BaseModel):
    """Resource configuration model."""
    cpu: Optional[str] = None
    memory: Optional[str] = None
    disk: Optional[str] = None
    network: Optional[Dict[str, Any]] = None
    env_vars: Optional[Dict[str, str]] = None
    labels: Optional[Dict[str, str]] = None
    annotations: Optional[Dict[str, str]] = None

    @validator('cpu')
    def validate_cpu(cls, v):
        if v and not re.match(r'^\d+[m]?$', v):
            raise ValueError('Invalid CPU format')
        return v

    @validator('memory')
    def validate_memory(cls, v):
        if v and not re.match(r'^\d+[KMGTPkmgtp]i?$', v):
            raise ValueError('Invalid memory format')
        return v

class ResourceStatus(BaseModel):
    """Resource status model."""
    state: str
    health: str
    last_update: datetime
    metrics: Optional[Dict[str, Any]] = None
    events: Optional[List[Dict[str, Any]]] = None

class ResourceMetrics(BaseModel):
    """Resource metrics model."""
    cpu_usage: float
    memory_usage: float
    network_in: float
    network_out: float
    disk_read: float
    disk_write: float
    timestamp: datetime

# Scaling Models
class ScalingRequest(BaseRequest):
    """Scaling request model."""
    resource_type: str
    resource_id: str
    target_state: Dict[str, Any]
    constraints: Optional[Dict[str, Any]] = None
    dry_run: bool = False

    @validator('resource_type')
    def validate_resource_type(cls, v):
        valid_types = {'deployment', 'statefulset', 'service', 'node'}
        if v not in valid_types:
            raise ValueError(f'Invalid resource type. Must be one of {valid_types}')
        return v

class ScalingResponse(BaseResponse):
    """Scaling response model."""
    operation_id: str
    resource_id: str
    current_state: Dict[str, Any]
    target_state: Dict[str, Any]
    validation_results: Optional[Dict[str, Any]] = None

# Pattern Models
class PatternDefinition(BaseModel):
    """Pattern definition model."""
    name: str
    description: Optional[str] = None
    version: str
    type: str
    parameters: Dict[str, Any]
    rules: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None

    @validator('version')
    def validate_version(cls, v):
        if not re.match(r'^\d+\.\d+\.\d+$', v):
            raise ValueError('Invalid version format')
        return v

class PatternInstance(BaseModel):
    """Pattern instance model."""
    pattern_id: str
    definition_id: str
    parameters: Dict[str, Any]
    state: Dict[str, Any]
    status: str
    created_at: datetime
    updated_at: datetime
    metadata: Optional[Dict[str, Any]] = None

# Container Models
class ContainerSpec(BaseModel):
    """Container specification model."""
    name: str
    image: str
    command: Optional[Union[str, List[str]]] = None
    args: Optional[List[str]] = None
    working_dir: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    ports: Optional[List[Dict[str, Any]]] = None
    volume_mounts: Optional[List[Dict[str, Any]]] = None
    resources: Optional[ResourceConfig] = None
    security_context: Optional[Dict[str, Any]] = None
    health_check: Optional[Dict[str, Any]] = None

    @validator('ports')
    def validate_ports(cls, v):
        if v:
            for port in v:
                if 'containerPort' not in port:
                    raise ValueError('Container port must be specified')
                if port['containerPort'] < 1 or port['containerPort'] > 65535:
                    raise ValueError('Invalid port number')
        return v

class ContainerState(BaseModel):
    """Container state model."""
    status: str
    running: Optional[Dict[str, Any]] = None
    terminated: Optional[Dict[str, Any]] = None
    waiting: Optional[Dict[str, Any]] = None
    last_state: Optional[Dict[str, Any]] = None
    ready: bool
    restart_count: int
    started: bool

# Rollback Models
class RollbackStep(BaseModel):
    """Rollback step model."""
    step_id: str
    action: str
    target: Dict[str, Any]
    parameters: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    timeout: Optional[int] = 300

class RollbackPlan(BaseModel):
    """Rollback plan model."""
    plan_id: str
    steps: List[RollbackStep]
    validation_mode: str = "strict"
    timeout: int = 3600
    metadata: Optional[Dict[str, Any]] = None

    @validator('steps')
    def validate_steps(cls, v):
        if not v:
            raise ValueError('At least one step is required')
        return v

# Event Models
class Event(BaseModel):
    """Event model."""
    event_id: str
    event_type: str
    source: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class EventFilter(BaseModel):
    """Event filter model."""
    event_types: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata_filters: Optional[Dict[str, Any]] = None

# Validation Models
class ValidationRule(BaseModel):
    """Validation rule model."""
    rule_id: str
    rule_type: str
    scope: str
    conditions: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None

class ValidationResult(BaseModel):
    """Validation result model."""
    is_valid: bool
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None

# Monitoring Models
class MetricsQuery(BaseModel):
    """Metrics query model."""
    resource_type: str
    resource_id: str
    metrics: List[str]
    start_time: datetime
    end_time: datetime
    interval: str
    aggregation: Optional[str] = None

    @validator('interval')
    def validate_interval(cls, v):
        if not re.match(r'^\d+[smhd]$', v):
            raise ValueError('Invalid interval format')
        return v

class AlertRule(BaseModel):
    """Alert rule model."""
    rule_id: str
    name: str
    condition: str
    threshold: float
    duration: str
    severity: str
    actions: List[Dict[str, Any]]
    enabled: bool = True

    @validator('severity')
    def validate_severity(cls, v):
        valid_severities = {'critical', 'warning', 'info'}
        if v not in valid_severities:
            raise ValueError(f'Invalid severity. Must be one of {valid_severities}')
        return v

# Response Models
class ErrorResponse(BaseResponse):
    """Error response model."""
    error_code: str
    error_details: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None

class ListResponse(BaseResponse):
    """List response model."""
    items: List[Dict[str, Any]]
    total_count: int
    page: Optional[int] = None
    page_size: Optional[int] = None
    next_token: Optional[str] = None

class HealthResponse(BaseResponse):
    """Health check response model."""
    services: Dict[str, bool]
    dependencies: Dict[str, bool]
    metrics: Dict[str, float]
    uptime: float