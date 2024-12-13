from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import asyncio
import logging
import json
import numpy as np
from enum import Enum
import jsonschema
from jsonschema import validators, Draft7Validator
import statistics
import hashlib
from collections import defaultdict

class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"
    NORMAL = "normal"
    LENIENT = "lenient"

class ValidationScope(Enum):
    """Validation scope types."""
    PATTERN = "pattern"
    OUTCOME = "outcome"
    COMPLETE = "complete"

@dataclass
class ValidationResult:
    """Container for validation results."""
    is_valid: bool
    confidence: float
    errors: List[str]
    warnings: List[str]
    metadata: Dict
    timestamp: datetime = field(default_factory=datetime.utcnow)
    validation_id: str = field(default_factory=lambda: hashlib.md5(str(datetime.utcnow().timestamp()).encode()).hexdigest())

class KnowledgeValidator:
    """
    Validates system knowledge patterns and outcomes.
    Implements multiple validation strategies and confidence scoring.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the knowledge validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.validation_level = ValidationLevel(
            config.get('validation_level', 'NORMAL')
        )
        
        # Load schemas
        self.schemas = self._load_validation_schemas()
        
        # Initialize validators
        self.validators = self._initialize_validators()
        
        # Statistical baselines
        self.statistical_baselines = self._initialize_statistical_baselines()
        
        # Validation history
        self.validation_history: List[ValidationResult] = []
        self.max_history_size = config.get('max_history_size', 1000)
        
        # Performance metrics
        self.performance_metrics = defaultdict(list)
        
        # Start background tasks
        asyncio.create_task(self._update_statistical_baselines_loop())
        asyncio.create_task(self._cleanup_history_loop())

    def _load_validation_schemas(self) -> Dict[str, Dict]:
        """Load JSON schemas for validation."""
        return {
            'scaling_pattern': {
                'type': 'object',
                'required': ['cpu_usage', 'memory_usage', 'request_rate'],
                'properties': {
                    'cpu_usage': {
                        'type': 'number',
                        'minimum': 0,
                        'maximum': 1
                    },
                    'memory_usage': {
                        'type': 'number',
                        'minimum': 0,
                        'maximum': 1
                    },
                    'request_rate': {
                        'type': 'number',
                        'minimum': 0
                    },
                    'error_rate': {
                        'type': 'number',
                        'minimum': 0,
                        'maximum': 1
                    }
                }
            },
            'scaling_outcome': {
                'type': 'object',
                'required': ['action', 'magnitude', 'success'],
                'properties': {
                    'action': {
                        'type': 'string',
                        'enum': ['scale_out', 'scale_in', 'scale_up', 'scale_down']
                    },
                    'magnitude': {
                        'type': 'number',
                        'minimum': 0
                    },
                    'success': {
                        'type': 'boolean'
                    },
                    'metrics': {
                        'type': 'object',
                        'properties': {
                            'response_time': {'type': 'number', 'minimum': 0},
                            'throughput': {'type': 'number', 'minimum': 0},
                            'error_rate': {'type': 'number', 'minimum': 0}
                        }
                    }
                }
            },
            'workload_pattern': {
                'type': 'object',
                'required': ['request_rate', 'pattern_type'],
                'properties': {
                    'request_rate': {
                        'type': 'array',
                        'items': {'type': 'number', 'minimum': 0}
                    },
                    'pattern_type': {
                        'type': 'string',
                        'enum': ['spike', 'gradual', 'periodic', 'random']
                    }
                }
            }
            # Add more schemas as needed
        }

    def _initialize_validators(self) -> Dict[str, Any]:
        """Initialize JSON schema validators."""
        return {
            name: Draft7Validator(schema)
            for name, schema in self.schemas.items()
        }

    def _initialize_statistical_baselines(self) -> Dict[str, Dict]:
        """Initialize statistical baselines for validation."""
        return {
            'cpu_usage': {
                'mean': 0.5,
                'std': 0.2,
                'min': 0,
                'max': 1
            },
            'memory_usage': {
                'mean': 0.6,
                'std': 0.15,
                'min': 0,
                'max': 1
            },
            'request_rate': {
                'mean': 100,
                'std': 50,
                'min': 0,
                'max': 1000
            },
            'error_rate': {
                'mean': 0.01,
                'std': 0.005,
                'min': 0,
                'max': 0.1
            }
        }

    async def validate_knowledge(
        self,
        pattern: Dict,
        outcome: Optional[Dict] = None,
        scope: ValidationScope = ValidationScope.COMPLETE,
        metadata: Optional[Dict] = None
    ) -> ValidationResult:
        """
        Validate knowledge pattern and/or outcome.
        
        Args:
            pattern: Pattern data to validate
            outcome: Optional outcome data to validate
            scope: Validation scope
            metadata: Optional metadata
            
        Returns:
            ValidationResult object
        """
        try:
            errors = []
            warnings = []
            
            # Validate pattern
            if scope in [ValidationScope.PATTERN, ValidationScope.COMPLETE]:
                pattern_errors, pattern_warnings = self._validate_pattern(pattern)
                errors.extend(pattern_errors)
                warnings.extend(pattern_warnings)
            
            # Validate outcome if provided
            if outcome and scope in [ValidationScope.OUTCOME, ValidationScope.COMPLETE]:
                outcome_errors, outcome_warnings = self._validate_outcome(outcome)
                errors.extend(outcome_errors)
                warnings.extend(outcome_warnings)
            
            # Validate pattern-outcome relationship if both provided
            if outcome and scope == ValidationScope.COMPLETE:
                rel_errors, rel_warnings = self._validate_pattern_outcome_relationship(
                    pattern,
                    outcome
                )
                errors.extend(rel_errors)
                warnings.extend(rel_warnings)
            
            # Calculate confidence
            confidence = self._calculate_validation_confidence(
                len(errors),
                len(warnings),
                pattern,
                outcome
            )
            
            # Create result
            result = ValidationResult(
                is_valid=len(errors) == 0,
                confidence=confidence,
                errors=errors,
                warnings=warnings,
                metadata=metadata or {}
            )
            
            # Store in history
            self._store_validation_result(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error validating knowledge: {str(e)}")
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                metadata=metadata or {}
            )

    def _validate_pattern(
        self,
        pattern: Dict
    ) -> Tuple[List[str], List[str]]:
        """Validate pattern data."""
        errors = []
        warnings = []
        
        # Schema validation
        validator = self.validators.get('scaling_pattern')
        if validator:
            for error in validator.iter_errors(pattern):
                errors.append(f"Schema validation error: {error.message}")
        
        # Statistical validation
        for field, value in pattern.items():
            if field in self.statistical_baselines:
                baseline = self.statistical_baselines[field]
                
                # Check if value is within expected range
                if not baseline['min'] <= value <= baseline['max']:
                    errors.append(
                        f"Value {value} for {field} is outside expected range "
                        f"[{baseline['min']}, {baseline['max']}]"
                    )
                
                # Check if value is within normal distribution
                z_score = abs((value - baseline['mean']) / baseline['std'])
                if z_score > 3:  # More than 3 standard deviations
                    warnings.append(
                        f"Value {value} for {field} is statistically unusual "
                        f"(z-score: {z_score:.2f})"
                    )
        
        # Relationship validation
        if 'cpu_usage' in pattern and 'memory_usage' in pattern:
            if pattern['cpu_usage'] > 0.9 and pattern['memory_usage'] < 0.1:
                warnings.append(
                    "Unusual relationship between CPU and memory usage"
                )
        
        return errors, warnings

    def _validate_outcome(
        self,
        outcome: Dict
    ) -> Tuple[List[str], List[str]]:
        """Validate outcome data."""
        errors = []
        warnings = []
        
        # Schema validation
        validator = self.validators.get('scaling_outcome')
        if validator:
            for error in validator.iter_errors(outcome):
                errors.append(f"Schema validation error: {error.message}")
        
        # Business logic validation
        if 'action' in outcome and 'magnitude' in outcome:
            action = outcome['action']
            magnitude = outcome['magnitude']
            
            # Check action-magnitude relationship
            if action in ['scale_out', 'scale_in'] and not float(magnitude).is_integer():
                errors.append(
                    f"Invalid magnitude {magnitude} for {action} action: "
                    "must be an integer"
                )
            
            # Check reasonable limits
            if action == 'scale_out' and magnitude > 10:
                warnings.append(
                    f"Unusually large scale out magnitude: {magnitude}"
                )
        
        # Performance metrics validation
        if 'metrics' in outcome:
            metrics = outcome['metrics']
            
            if 'response_time' in metrics:
                if metrics['response_time'] > 1000:  # ms
                    warnings.append(
                        f"High response time: {metrics['response_time']}ms"
                    )
            
            if 'error_rate' in metrics:
                if metrics['error_rate'] > 0.05:  # 5%
                    warnings.append(
                        f"High error rate: {metrics['error_rate']*100}%"
                    )
        
        return errors, warnings

    def _validate_pattern_outcome_relationship(
        self,
        pattern: Dict,
        outcome: Dict
    ) -> Tuple[List[str], List[str]]:
        """Validate relationship between pattern and outcome."""
        errors = []
        warnings = []
        
        # Validate scaling decisions
        if 'action' in outcome:
            action = outcome['action']
            
            if 'cpu_usage' in pattern and 'memory_usage' in pattern:
                cpu_usage = pattern['cpu_usage']
                memory_usage = pattern['memory_usage']
                
                # Check if scaling action matches resource usage
                if action == 'scale_out':
                    if max(cpu_usage, memory_usage) < 0.5:
                        warnings.append(
                            f"Scale out action with low resource usage "
                            f"(CPU: {cpu_usage:.2f}, Memory: {memory_usage:.2f})"
                        )
                elif action == 'scale_in':
                    if max(cpu_usage, memory_usage) > 0.7:
                        warnings.append(
                            f"Scale in action with high resource usage "
                            f"(CPU: {cpu_usage:.2f}, Memory: {memory_usage:.2f})"
                        )
        
        # Validate performance impact
        if 'metrics' in outcome and 'request_rate' in pattern:
            metrics = outcome['metrics']
            request_rate = pattern['request_rate']
            
            if 'throughput' in metrics:
                throughput = metrics['throughput']
                
                # Check if throughput matches request rate
                if throughput < request_rate * 0.8:
                    warnings.append(
                        f"Throughput ({throughput}) is significantly lower than "
                        f"request rate ({request_rate})"
                    )
        
        return errors, warnings

    def _calculate_validation_confidence(
        self,
        error_count: int,
        warning_count: int,
        pattern: Dict,
        outcome: Optional[Dict]
    ) -> float:
        """Calculate confidence score for validation."""
        try:
            # Base confidence
            if error_count > 0:
                return 0.0
            
            base_confidence = 1.0 - (warning_count * 0.1)  # Reduce by 0.1 for each warning
            
            # Pattern confidence
            pattern_confidence = self._calculate_pattern_confidence(pattern)
            
            # Outcome confidence if available
            outcome_confidence = 1.0
            if outcome:
                outcome_confidence = self._calculate_outcome_confidence(outcome)
            
            # Combine confidences
            final_confidence = (
                base_confidence * 0.4 +
                pattern_confidence * 0.3 +
                outcome_confidence * 0.3
            )
            
            return max(0, min(1, final_confidence))
            
        except Exception:
            return 0.0

    def _calculate_pattern_confidence(
        self,
        pattern: Dict
    ) -> float:
        """Calculate confidence score for pattern."""
        try:
            confidences = []
            
            # Calculate confidence for each metric
            for field, value in pattern.items():
                if field in self.statistical_baselines:
                    baseline = self.statistical_baselines[field]
                    
                    # Calculate z-score
                    z_score = abs((value - baseline['mean']) / baseline['std'])
                    
                    # Convert to confidence score
                    confidence = 1.0 / (1.0 + z_score)
                    confidences.append(confidence)
            
            return statistics.mean(confidences) if confidences else 0.5
            
        except Exception:
            return 0.5

    def _calculate_outcome_confidence(
        self,
        outcome: Dict
    ) -> float:
        """Calculate confidence score for outcome."""
        try:
            factors = []
            
            # Success factor
            if outcome.get('success', False):
                factors.append(1.0)
            else:
                factors.append(0.3)
            
            # Performance metrics factor
            if 'metrics' in outcome:
                metrics = outcome['metrics']
                
                if 'response_time' in metrics:
                    rt_confidence = 1.0 / (1.0 + metrics['response_time'] / 1000)
                    factors.append(rt_confidence)
                
                if 'error_rate' in metrics:
                    error_confidence = 1.0 - metrics['error_rate']
                    factors.append(error_confidence)
            
            return statistics.mean(factors) if factors else 0.5
            
        except Exception:
            return 0.5

    def _store_validation_result(
        self,
        result: ValidationResult
    ):
        """Store validation result in history."""
        self.validation_history.append(result)
        
        # Trim history if needed
        if len(self.validation_history) > self.max_history_size:
            self.validation_history = self.validation_history[-self.max_history_size:]
        
        # Update performance metrics
        self.performance_metrics['error_count'].append(len(result.errors))
        self.performance_metrics['warning_count'].append(len(result.warnings))
        self.performance_metrics['confidence'].append(result.confidence)

    async def _update_statistical_baselines_loop(self):
        """Background task for updating statistical baselines."""
        while True:
            try:
                # Update baselines based on recent validation history
                patterns = []
                for result in self.validation_history[-1000:]:  # Last 1000 validations
                    if result.is_valid and result.confidence > 0.8:
                        if 'pattern' in result.metadata:
                            patterns.append(result.metadata['pattern'])
                
                if patterns:
                    # Update statistical baselines
                    for field in self.statistical_baselines:
                        values = [p[field] for p in patterns if field in p]
                        if values:
                            self.statistical_baselines[field] = {
                                'mean': statistics.mean(values),
                                'std': statistics.stdev(values) if len(values) > 1 else 0.1,
                                'min': min(values),
                                'max': max(values)
                            }
                
                # Sleep before next update
                await asyncio.sleep(
                    self.config.get('baseline_update_interval', 3600)  # 1 hour
                )
                
            except Exception as e:
                self.logger.error(f"Error updating baselines: {str(e)}")
                await asyncio.sleep(60)

    async def _cleanup_history_loop(self):
        """Background task for cleaning up validation history."""
        while True:
            try:
                # Get retention period
                retention_hours = self.config.get('history_retention_hours', 24)
                cutoff_time = datetime.utcnow() - timedelta(hours=retention_hours)
                
                # Clean up old entries
                self.validation_history = [
                    result for result in self.validation_history
                    if result.timestamp > cutoff_time
                ]
                
                # Sleep before next cleanup
                await asyncio.sleep(
                    self.config.get('cleanup_interval', 3600)  # 1 hour
                )
                
            except Exception as e:
                self.logger.error(f"Error in history cleanup: {str(e)}")
                await asyncio.sleep(60)

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get validator performance metrics."""
        try:
            recent_metrics = {
                'validation_count': len(self.validation_history),
                'error_rate': statistics.mean(self.performance_metrics['error_count'][-100:]),
                'warning_rate': statistics.mean(self.performance_metrics['warning_count'][-100:]),
                'average_confidence': statistics.mean(self.performance_metrics['confidence'][-100:])
            }
            return recent_metrics
        except Exception:
            return {
                'validation_count': 0,
                'error_rate': 0,
                'warning_rate': 0,
                'average_confidence': 0
            }

# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'validation_level': 'NORMAL',
        'max_history_size': 1000,
        'baseline_update_interval': 3600,
        'history_retention_hours': 24
    }
    
    # Initialize validator
    validator = KnowledgeValidator(config)
    
    # Example validation
    async def main():
        # Example pattern
        pattern = {
            'cpu_usage': 0.8,
            'memory_usage': 0.7,
            'request_rate': 100,
            'error_rate': 0.01
        }
        
        # Example outcome
        outcome = {
            'action': 'scale_out',
            'magnitude': 2,
            'success': True,
            'metrics': {
                'response_time': 200,
                'throughput': 90,
                'error_rate': 0.02
            }
        }
        
        # Validate knowledge
        result = await validator.validate_knowledge(
            pattern,
            outcome,
            ValidationScope.COMPLETE,
            {'pattern': pattern, 'source': 'auto_scaler'}
        )
        
        print(f"Validation Result:")
        print(f"Valid: {result.is_valid}")
        print(f"Confidence: {result.confidence:.2f}")
        print("\nErrors:")
        for error in result.errors:
            print(f"- {error}")
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"- {warning}")
        
        # Get performance metrics
        metrics = validator.get_performance_metrics()
        print("\nValidator Performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")
    
    # Run example
    asyncio.run(main())