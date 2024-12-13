from typing import Optional, Any
from datetime import datetime

class EffiScaleException(Exception):
    """Base exception class for all EffiScale exceptions."""
    
    def __init__(self, message: str, error_code: str, details: Optional[dict] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.utcnow()
        super().__init__(self.message)

    def to_dict(self) -> dict:
        """Convert exception to dictionary format for logging and API responses."""
        return {
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'type': self.__class__.__name__
        }

# Monitoring Exceptions
class MonitoringException(EffiScaleException):
    """Base class for monitoring-related exceptions."""
    pass

class MetricCollectionError(MonitoringException):
    """Raised when metric collection fails."""
    def __init__(self, message: str, container_id: str, metric_type: str):
        super().__init__(
            message=message,
            error_code='METRIC_COLLECTION_ERROR',
            details={
                'container_id': container_id,
                'metric_type': metric_type
            }
        )

class PrometheusConnectionError(MonitoringException):
    """Raised when connection to Prometheus fails."""
    def __init__(self, message: str, endpoint: str):
        super().__init__(
            message=message,
            error_code='PROMETHEUS_CONNECTION_ERROR',
            details={'endpoint': endpoint}
        )

# Analysis Exceptions
class AnalysisException(EffiScaleException):
    """Base class for analysis-related exceptions."""
    pass

class PatternAnalysisError(AnalysisException):
    """Raised when pattern analysis fails."""
    def __init__(self, message: str, pattern_id: Optional[str] = None):
        super().__init__(
            message=message,
            error_code='PATTERN_ANALYSIS_ERROR',
            details={'pattern_id': pattern_id}
        )

class ThresholdCalculationError(AnalysisException):
    """Raised when threshold calculation fails."""
    def __init__(self, message: str, metric_type: str, current_value: Any):
        super().__init__(
            message=message,
            error_code='THRESHOLD_CALCULATION_ERROR',
            details={
                'metric_type': metric_type,
                'current_value': current_value
            }
        )

# Controller Exceptions
class ControllerException(EffiScaleException):
    """Base class for controller-related exceptions."""
    pass

class ScalingDecisionError(ControllerException):
    """Raised when scaling decision making fails."""
    def __init__(self, message: str, metrics: dict):
        super().__init__(
            message=message,
            error_code='SCALING_DECISION_ERROR',
            details={'metrics': metrics}
        )

class ConsensusError(ControllerException):
    """Raised when controller consensus cannot be reached."""
    def __init__(self, message: str, controller_ids: list):
        super().__init__(
            message=message,
            error_code='CONSENSUS_ERROR',
            details={'controller_ids': controller_ids}
        )

# Execution Exceptions
class ExecutionException(EffiScaleException):
    """Base class for execution-related exceptions."""
    pass

class ScalingExecutionError(ExecutionException):
    """Raised when scaling execution fails."""
    def __init__(self, message: str, scaling_type: str, target_state: dict):
        super().__init__(
            message=message,
            error_code='SCALING_EXECUTION_ERROR',
            details={
                'scaling_type': scaling_type,
                'target_state': target_state
            }
        )

class RollbackError(ExecutionException):
    """Raised when scaling rollback fails."""
    def __init__(self, message: str, original_state: dict):
        super().__init__(
            message=message,
            error_code='ROLLBACK_ERROR',
            details={'original_state': original_state}
        )

# Knowledge Base Exceptions
class KnowledgeBaseException(EffiScaleException):
    """Base class for knowledge base-related exceptions."""
    pass

class KnowledgeUpdateError(KnowledgeBaseException):
    """Raised when knowledge base update fails."""
    def __init__(self, message: str, update_type: str, data: Any):
        super().__init__(
            message=message,
            error_code='KNOWLEDGE_UPDATE_ERROR',
            details={
                'update_type': update_type,
                'data': str(data)
            }
        )

class ValidationError(KnowledgeBaseException):
    """Raised when knowledge validation fails."""
    def __init__(self, message: str, validation_type: str, invalid_data: Any):
        super().__init__(
            message=message,
            error_code='VALIDATION_ERROR',
            details={
                'validation_type': validation_type,
                'invalid_data': str(invalid_data)
            }
        )

# Configuration Exceptions
class ConfigurationException(EffiScaleException):
    """Base class for configuration-related exceptions."""
    pass

class ConfigurationLoadError(ConfigurationException):
    """Raised when configuration loading fails."""
    def __init__(self, message: str, config_path: str):
        super().__init__(
            message=message,
            error_code='CONFIG_LOAD_ERROR',
            details={'config_path': config_path}
        )

class ConfigurationValidationError(ConfigurationException):
    """Raised when configuration validation fails."""
    def __init__(self, message: str, invalid_keys: list):
        super().__init__(
            message=message,
            error_code='CONFIG_VALIDATION_ERROR',
            details={'invalid_keys': invalid_keys}
        )

# Storage Exceptions
class StorageException(EffiScaleException):
    """Base class for storage-related exceptions."""
    pass

class MetricStorageError(StorageException):
    """Raised when metric storage operations fail."""
    def __init__(self, message: str, operation: str, metric_data: Optional[dict] = None):
        super().__init__(
            message=message,
            error_code='METRIC_STORAGE_ERROR',
            details={
                'operation': operation,
                'metric_data': metric_data
            }
        )

class DatabaseConnectionError(StorageException):
    """Raised when database connection fails."""
    def __init__(self, message: str, db_url: str):
        super().__init__(
            message=message,
            error_code='DB_CONNECTION_ERROR',
            details={'db_url': db_url}
        )

def handle_exception(exception: EffiScaleException) -> dict:
    """
    Global exception handler that processes all EffiScale exceptions.
    
    Args:
        exception: The caught exception
        
    Returns:
        dict: Formatted error response
    """
    error_info = exception.to_dict()

    
    return {
        'status': 'error',
        'error': error_info
    }