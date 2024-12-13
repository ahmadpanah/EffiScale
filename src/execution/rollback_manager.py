from typing import Dict, List, Optional, Tuple, Union, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import asyncio
import logging
import json
from enum import Enum
import hashlib
from collections import defaultdict
import aioredis
import aiosqlite
import pickle
import copy
import traceback

class RollbackType(Enum):
    """Types of rollback operations."""
    CONFIGURATION = "configuration"
    DEPLOYMENT = "deployment"
    SCALING = "scaling"
    DATABASE = "database"
    NETWORK = "network"
    COMPLETE = "complete"

class RollbackStatus(Enum):
    """Status of rollback operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_COMPLETED = "partially_completed"
    CANCELLED = "cancelled"

@dataclass
class RollbackStep:
    """Container for rollback steps."""
    step_id: str
    action: str
    target: Dict
    state_before: Dict
    state_after: Optional[Dict] = None
    status: RollbackStatus = RollbackStatus.PENDING
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration: Optional[float] = None

@dataclass
class RollbackOperation:
    """Container for rollback operations."""
    operation_id: str
    rollback_type: RollbackType
    steps: List[RollbackStep]
    status: RollbackStatus
    metadata: Dict
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    triggered_by: Optional[str] = None
    error: Optional[str] = None

class RollbackManager:
    """
    Manages system rollbacks and recovery operations.
    Implements rollback strategies and state management.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the rollback manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.operations: Dict[str, RollbackOperation] = {}
        self.pending_operations: List[str] = []
        self.active_operations: List[str] = []
        
        # State management
        self.state_store: Dict[str, Any] = {}
        self.state_history: List[Dict] = []
        
        # Redis connection
        self.redis_pool = None
        
        # SQLite connection
        self.sqlite_conn = None
        
        # Operation settings
        self.max_concurrent_operations = config.get('max_concurrent_operations', 3)
        self.operation_timeout = config.get('operation_timeout', 300)
        self.max_retry_attempts = config.get('max_retry_attempts', 3)
        
        # State management settings
        self.max_state_history = config.get('max_state_history', 100)
        self.state_retention_days = config.get('state_retention_days', 30)
        
        # Start background tasks
        asyncio.create_task(self._initialize_storage())
        asyncio.create_task(self._operation_processor_loop())
        asyncio.create_task(self._state_cleanup_loop())

    async def create_rollback(
        self,
        rollback_type: RollbackType,
        steps: List[Dict],
        metadata: Optional[Dict] = None,
        triggered_by: Optional[str] = None
    ) -> RollbackOperation:
        """
        Create new rollback operation.
        
        Args:
            rollback_type: Type of rollback
            steps: List of rollback steps
            metadata: Optional metadata
            triggered_by: Optional trigger identifier
            
        Returns:
            RollbackOperation object
        """
        try:
            # Generate operation ID
            operation_id = self._generate_operation_id(rollback_type, steps)
            
            # Create rollback steps
            rollback_steps = []
            for step in steps:
                # Get current state before rollback
                state_before = await self._get_current_state(
                    step['target']
                )
                
                rollback_steps.append(RollbackStep(
                    step_id=self._generate_step_id(operation_id, step),
                    action=step['action'],
                    target=step['target'],
                    state_before=state_before
                ))
            
            # Create operation
            operation = RollbackOperation(
                operation_id=operation_id,
                rollback_type=rollback_type,
                steps=rollback_steps,
                status=RollbackStatus.PENDING,
                metadata=metadata or {},
                triggered_by=triggered_by
            )
            
            # Store operation
            self.operations[operation_id] = operation
            self.pending_operations.append(operation_id)
            
            # Broadcast operation creation
            await self._broadcast_rollback_event(
                'rollback_created',
                operation
            )
            
            # Store initial state
            await self._store_state_snapshot(operation)
            
            return operation
            
        except Exception as e:
            self.logger.error(f"Error creating rollback: {str(e)}")
            raise

    async def execute_rollback(
        self,
        operation_id: str
    ) -> bool:
        """Execute rollback operation."""
        try:
            operation = self.operations.get(operation_id)
            if not operation:
                raise ValueError(f"Operation not found: {operation_id}")
            
            if operation.status != RollbackStatus.PENDING:
                raise ValueError(f"Invalid operation status: {operation.status}")
            
            # Move to active operations
            if operation_id in self.pending_operations:
                self.pending_operations.remove(operation_id)
            self.active_operations.append(operation_id)
            
            # Update status
            operation.status = RollbackStatus.IN_PROGRESS
            await self._broadcast_rollback_event(
                'rollback_started',
                operation
            )
            
            # Execute steps
            success = True
            for step in operation.steps:
                step_success = await self._execute_rollback_step(
                    operation,
                    step
                )
                if not step_success:
                    success = False
                    break
            
            # Update operation status
            if success:
                operation.status = RollbackStatus.COMPLETED
            else:
                operation.status = RollbackStatus.FAILED
                await self._handle_rollback_failure(operation)
            
            operation.completed_at = datetime.utcnow()
            
            # Remove from active operations
            if operation_id in self.active_operations:
                self.active_operations.remove(operation_id)
            
            # Broadcast completion
            await self._broadcast_rollback_event(
                'rollback_completed',
                operation
            )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error executing rollback: {str(e)}")
            return False

    async def cancel_rollback(
        self,
        operation_id: str
    ) -> bool:
        """Cancel pending rollback operation."""
        try:
            operation = self.operations.get(operation_id)
            if not operation:
                raise ValueError(f"Operation not found: {operation_id}")
            
            if operation.status != RollbackStatus.PENDING:
                raise ValueError(f"Cannot cancel operation in state: {operation.status}")
            
            # Update status
            operation.status = RollbackStatus.CANCELLED
            
            # Remove from pending operations
            if operation_id in self.pending_operations:
                self.pending_operations.remove(operation_id)
            
            # Broadcast cancellation
            await self._broadcast_rollback_event(
                'rollback_cancelled',
                operation
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling rollback: {str(e)}")
            return False

    async def get_rollback_status(
        self,
        operation_id: str
    ) -> Optional[RollbackStatus]:
        """Get status of rollback operation."""
        try:
            operation = self.operations.get(operation_id)
            return operation.status if operation else None
        except Exception as e:
            self.logger.error(f"Error getting rollback status: {str(e)}")
            return None

    async def get_rollback_history(
        self,
        rollback_type: Optional[RollbackType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[RollbackOperation]:
        """Get rollback operation history."""
        try:
            operations = []
            
            # Filter operations
            for op in self.operations.values():
                if rollback_type and op.rollback_type != rollback_type:
                    continue
                    
                if start_time and op.created_at < start_time:
                    continue
                    
                if end_time and op.created_at > end_time:
                    continue
                    
                operations.append(op)
            
            # Sort by creation time
            operations.sort(key=lambda x: x.created_at, reverse=True)
            
            if limit:
                operations = operations[:limit]
            
            return operations
            
        except Exception as e:
            self.logger.error(f"Error getting rollback history: {str(e)}")
            return []

    async def _execute_rollback_step(
        self,
        operation: RollbackOperation,
        step: RollbackStep
    ) -> bool:
        """Execute single rollback step."""
        try:
            start_time = datetime.utcnow()
            
            # Update step status
            step.status = RollbackStatus.IN_PROGRESS
            
            # Execute step action
            if step.action == 'restore_configuration':
                success = await self._restore_configuration(step)
            elif step.action == 'rollback_deployment':
                success = await self._rollback_deployment(step)
            elif step.action == 'revert_scaling':
                success = await self._revert_scaling(step)
            elif step.action == 'restore_database':
                success = await self._restore_database(step)
            elif step.action == 'reset_network':
                success = await self._reset_network(step)
            else:
                raise ValueError(f"Unsupported action: {step.action}")
            
            # Update step status and state
            step.status = RollbackStatus.COMPLETED if success else RollbackStatus.FAILED
            step.state_after = await self._get_current_state(step.target)
            step.duration = (datetime.utcnow() - start_time).total_seconds()
            
            # Store state change
            await self._store_state_change(operation, step)
            
            return success
            
        except Exception as e:
            step.status = RollbackStatus.FAILED
            step.error = str(e)
            self.logger.error(f"Error executing rollback step: {str(e)}")
            return False

    async def _restore_configuration(
        self,
        step: RollbackStep
    ) -> bool:
        """Restore system configuration."""
        try:
            target = step.target
            config_path = target.get('path')
            config_type = target.get('type')
            
            # Get configuration backup
            backup = await self._get_configuration_backup(
                config_path,
                config_type
            )
            
            if not backup:
                raise ValueError("Configuration backup not found")
            
            # Apply configuration
            if config_type == 'yaml':
                success = await self._apply_yaml_configuration(
                    config_path,
                    backup
                )
            elif config_type == 'json':
                success = await self._apply_json_configuration(
                    config_path,
                    backup
                )
            elif config_type == 'env':
                success = await self._apply_env_configuration(
                    config_path,
                    backup
                )
            else:
                raise ValueError(f"Unsupported configuration type: {config_type}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error restoring configuration: {str(e)}")
            return False

    async def _rollback_deployment(
        self,
        step: RollbackStep
    ) -> bool:
        """Rollback deployment to previous version."""
        try:
            target = step.target
            deployment_type = target.get('type')
            deployment_id = target.get('id')
            
            # Get previous version
            previous_version = await self._get_previous_version(
                deployment_type,
                deployment_id
            )
            
            if not previous_version:
                raise ValueError("Previous version not found")
            
            # Execute rollback
            if deployment_type == 'kubernetes':
                success = await self._rollback_kubernetes_deployment(
                    deployment_id,
                    previous_version
                )
            elif deployment_type == 'docker':
                success = await self._rollback_docker_deployment(
                    deployment_id,
                    previous_version
                )
            else:
                raise ValueError(f"Unsupported deployment type: {deployment_type}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error rolling back deployment: {str(e)}")
            return False

    async def _revert_scaling(
        self,
        step: RollbackStep
    ) -> bool:
        """Revert scaling operation."""
        try:
            target = step.target
            resource_type = target.get('type')
            resource_id = target.get('id')
            
            # Get previous scale
            previous_scale = await self._get_previous_scale(
                resource_type,
                resource_id
            )
            
            if not previous_scale:
                raise ValueError("Previous scale not found")
            
            # Execute scaling
            if resource_type == 'kubernetes':
                success = await self._scale_kubernetes_resource(
                    resource_id,
                    previous_scale
                )
            else:
                raise ValueError(f"Unsupported resource type: {resource_type}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error reverting scaling: {str(e)}")
            return False

    async def _handle_rollback_failure(
        self,
        operation: RollbackOperation
    ):
        """Handle rollback operation failure."""
        try:
            self.logger.error(f"Rollback operation failed: {operation.operation_id}")
            
            # Collect failure information
            failed_steps = [
                step for step in operation.steps
                if step.status == RollbackStatus.FAILED
            ]
            
            # Store failure details
            operation.error = "\n".join([
                f"Step {step.step_id} failed: {step.error}"
                for step in failed_steps
            ])
            
            # Notify about failure
            await self._broadcast_rollback_event(
                'rollback_failed',
                operation
            )
            
            # Store failure state
            await self._store_failure_state(operation)
            
        except Exception as e:
            self.logger.error(f"Error handling rollback failure: {str(e)}")

    async def _store_state_snapshot(
        self,
        operation: RollbackOperation
    ):
        """Store system state snapshot."""
        try:
            # Prepare state snapshot
            snapshot = {
                'timestamp': datetime.utcnow().isoformat(),
                'operation_id': operation.operation_id,
                'states': {}
            }
            
            # Collect states for all steps
            for step in operation.steps:
                snapshot['states'][step.step_id] = {
                    'before': step.state_before,
                    'after': step.state_after
                }
            
            # Store in history
            self.state_history.append(snapshot)
            
            # Trim history if needed
            if len(self.state_history) > self.max_state_history:
                self.state_history = self.state_history[-self.max_state_history:]
            
            # Store in SQLite
            if self.sqlite_conn:
                await self.sqlite_conn.execute("""
                    INSERT INTO state_history (
                        timestamp,
                        operation_id,
                        state_data
                    ) VALUES (?, ?, ?)
                """, (
                    datetime.utcnow().isoformat(),
                    operation.operation_id,
                    json.dumps(snapshot)
                ))
                await self.sqlite_conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error storing state snapshot: {str(e)}")

    def _generate_operation_id(
        self,
        rollback_type: RollbackType,
        steps: List[Dict]
    ) -> str:
        """Generate unique operation ID."""
        data = f"{rollback_type.value}:{json.dumps(steps)}:{datetime.utcnow().isoformat()}"
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    def _generate_step_id(
        self,
        operation_id: str,
        step: Dict
    ) -> str:
        """Generate unique step ID."""
        data = f"{operation_id}:{json.dumps(step)}"
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    async def _broadcast_rollback_event(
        self,
        event_type: str,
        operation: RollbackOperation
    ):
        """Broadcast rollback event."""
        try:
            if self.redis_pool:
                event = {
                    'event_type': event_type,
                    'operation_id': operation.operation_id,
                    'rollback_type': operation.rollback_type.value,
                    'status': operation.status.value,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                await self.redis_pool.publish(
                    'rollback_events',
                    json.dumps(event)
                )
            
        except Exception as e:
            self.logger.error(f"Error broadcasting event: {str(e)}")

    async def _operation_processor_loop(self):
        """Background task for processing rollback operations."""
        while True:
            try:
                # Process pending operations
                while len(self.active_operations) < self.max_concurrent_operations and \
                      self.pending_operations:
                    operation_id = self.pending_operations.pop(0)
                    asyncio.create_task(self.execute_rollback(operation_id))
                
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in operation processor: {str(e)}")
                await asyncio.sleep(5)

    async def _state_cleanup_loop(self):
        """Background task for cleaning up old state history."""
        while True:
            try:
                # Clean up old state history
                retention_date = datetime.utcnow() - timedelta(
                    days=self.state_retention_days
                )
                
                if self.sqlite_conn:
                    await self.sqlite_conn.execute("""
                        DELETE FROM state_history
                        WHERE timestamp < ?
                    """, (retention_date.isoformat(),))
                    await self.sqlite_conn.commit()
                
                # Clean up memory state history
                self.state_history = [
                    state for state in self.state_history
                    if datetime.fromisoformat(state['timestamp']) > retention_date
                ]
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                self.logger.error(f"Error in state cleanup: {str(e)}")
                await asyncio.sleep(300)

    async def _initialize_storage(self):
        """Initialize storage backends."""
        try:
            # Initialize SQLite
            if 'sqlite_path' in self.config:
                self.sqlite_conn = await aiosqlite.connect(
                    self.config['sqlite_path']
                )
                
                # Create tables
                await self.sqlite_conn.execute("""
                    CREATE TABLE IF NOT EXISTS state_history (
                        timestamp TEXT,
                        operation_id TEXT,
                        state_data TEXT
                    )
                """)
                await self.sqlite_conn.commit()
            
            # Initialize Redis
            if 'redis_url' in self.config:
                self.redis_pool = await aioredis.create_redis_pool(
                    self.config['redis_url'],
                    minsize=5,
                    maxsize=10
                )
            
        except Exception as e:
            self.logger.error(f"Error initializing storage: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'sqlite_path': 'rollback.db',
        'redis_url': 'redis://localhost',
        'max_concurrent_operations': 3,
        'operation_timeout': 300,
        'max_retry_attempts': 3,
        'max_state_history': 100,
        'state_retention_days': 30
    }
    
    # Initialize manager
    manager = RollbackManager(config)
    
    # Example rollback operation
    async def main():
        # Create rollback operation
        operation = await manager.create_rollback(
            rollback_type=RollbackType.DEPLOYMENT,
            steps=[
                {
                    'action': 'rollback_deployment',
                    'target': {
                        'type': 'kubernetes',
                        'id': 'my-deployment',
                        'namespace': 'default'
                    }
                },
                {
                    'action': 'revert_scaling',
                    'target': {
                        'type': 'kubernetes',
                        'id': 'my-deployment',
                        'namespace': 'default'
                    }
                }
            ],
            metadata={'environment': 'production'},
            triggered_by='deploy_pipeline'
        )
        
        print(f"Created rollback operation: {operation.operation_id}")
        
        # Execute rollback
        success = await manager.execute_rollback(operation.operation_id)
        print(f"Rollback {'succeeded' if success else 'failed'}")
        
        # Get operation status
        status = await manager.get_rollback_status(operation.operation_id)
        print(f"Final status: {status.value}")
        
        # Get rollback history
        history = await manager.get_rollback_history(
            rollback_type=RollbackType.DEPLOYMENT,
            limit=5
        )
        
        print("\nRecent rollback history:")
        for op in history:
            print(f"Operation {op.operation_id}: {op.status.value}")
    
    # Run example
    asyncio.run(main())