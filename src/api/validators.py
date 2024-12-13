from typing import Dict, List, Optional, Set, Any, Union, Tuple
from datetime import datetime
from pydantic import BaseModel, ValidationError, validator
import re
import json
import asyncio
import logging
from enum import Enum
from dataclasses import dataclass
import yaml
import jsonschema
from jsonschema import validate, ValidationError as JsonSchemaError

class ValidationScope(Enum):
    """Validation scopes."""
    CONFIGURATION = "configuration"
    PATTERN = "pattern"
    DEPLOYMENT = "deployment"
    SCALING = "scaling"
    SECURITY = "security"
    COMPLIANCE = "compliance"

class ValidationLevel(Enum):
    """Validation levels."""
    STRICT = "strict"
    NORMAL = "normal"
    RELAXED = "relaxed"

class ValidationPriority(Enum):
    """Validation priority levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class ValidationResult:
    """Container for validation results."""
    is_valid: bool
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: datetime = datetime.utcnow()

class KnowledgeValidator:
    """
    Validates input against knowledge base and rules.
    Implements validation strategies and rule checking.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.validation_rules: Dict[str, Dict] = {}
        self.schema_cache: Dict[str, Dict] = {}
        self.validation_history: List[Dict] = []
        
        # Load configuration
        self.validation_level = ValidationLevel(
            config.get('validation_level', 'NORMAL')
        )
        self.max_history_size = config.get('max_history_size', 1000)
        
        # Load rules and schemas
        self._load_validation_rules()
        self._load_json_schemas()

    async def validate_knowledge(
        self,
        data: Dict,
        scope: ValidationScope,
        level: Optional[ValidationLevel] = None,
        metadata: Optional[Dict] = None
    ) -> ValidationResult:
        """
        Validate data against knowledge base.
        
        Args:
            data: Data to validate
            scope: Validation scope
            level: Optional validation level override
            metadata: Optional metadata
            
        Returns:
            ValidationResult object
        """
        try:
            validation_level = level or self.validation_level
            errors = []
            warnings = []
            
            # Schema validation
            schema_errors = await self._validate_schema(data, scope)
            errors.extend(schema_errors)
            
            # Rule validation
            rule_errors, rule_warnings = await self._validate_rules(
                data,
                scope,
                validation_level
            )
            errors.extend(rule_errors)
            warnings.extend(rule_warnings)
            
            # Domain validation
            domain_errors, domain_warnings = await self._validate_domain(
                data,
                scope,
                validation_level
            )
            errors.extend(domain_errors)
            warnings.extend(domain_warnings)
            
            # Create result
            result = ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                metadata=metadata or {}
            )
            
            # Store in history
            await self._store_validation_result(result, data, scope)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            return ValidationResult(
                is_valid=False,
                errors=[{"error": str(e), "type": "system"}],
                warnings=[],
                metadata=metadata or {}
            )

    async def validate_pattern(
        self,
        pattern: Dict,
        metadata: Optional[Dict] = None
    ) -> ValidationResult:
        """Validate pattern definition."""
        try:
            errors = []
            warnings = []
            
            # Validate pattern structure
            structure_errors = await self._validate_pattern_structure(pattern)
            errors.extend(structure_errors)
            
            # Validate pattern logic
            logic_errors, logic_warnings = await self._validate_pattern_logic(pattern)
            errors.extend(logic_errors)
            warnings.extend(logic_warnings)
            
            # Validate pattern parameters
            param_errors, param_warnings = await self._validate_pattern_parameters(
                pattern
            )
            errors.extend(param_errors)
            warnings.extend(param_warnings)
            
            # Create result
            result = ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                metadata=metadata or {}
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pattern validation error: {str(e)}")
            return ValidationResult(
                is_valid=False,
                errors=[{"error": str(e), "type": "pattern"}],
                warnings=[],
                metadata=metadata or {}
            )

    async def validate_configuration(
        self,
        config: Dict,
        config_type: str,
        metadata: Optional[Dict] = None
    ) -> ValidationResult:
        """Validate configuration data."""
        try:
            errors = []
            warnings = []
            
            # Validate configuration structure
            structure_errors = await self._validate_config_structure(
                config,
                config_type
            )
            errors.extend(structure_errors)
            
            # Validate configuration values
            value_errors, value_warnings = await self._validate_config_values(
                config,
                config_type
            )
            errors.extend(value_errors)
            warnings.extend(value_warnings)
            
            # Validate configuration dependencies
            dep_errors, dep_warnings = await self._validate_config_dependencies(
                config,
                config_type
            )
            errors.extend(dep_errors)
            warnings.extend(dep_warnings)
            
            # Create result
            result = ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                metadata=metadata or {}
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Configuration validation error: {str(e)}")
            return ValidationResult(
                is_valid=False,
                errors=[{"error": str(e), "type": "configuration"}],
                warnings=[],
                metadata=metadata or {}
            )

    async def validate_security(
        self,
        data: Dict,
        security_level: str,
        metadata: Optional[Dict] = None
    ) -> ValidationResult:
        """Validate security requirements."""
        try:
            errors = []
            warnings = []
            
            # Validate security policies
            policy_errors = await self._validate_security_policies(
                data,
                security_level
            )
            errors.extend(policy_errors)
            
            # Validate access controls
            access_errors, access_warnings = await self._validate_access_controls(
                data,
                security_level
            )
            errors.extend(access_errors)
            warnings.extend(access_warnings)
            
            # Validate encryption requirements
            encryption_errors = await self._validate_encryption(
                data,
                security_level
            )
            errors.extend(encryption_errors)
            
            # Create result
            result = ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                metadata=metadata or {}
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Security validation error: {str(e)}")
            return ValidationResult(
                is_valid=False,
                errors=[{"error": str(e), "type": "security"}],
                warnings=[],
                metadata=metadata or {}
            )

    async def _validate_schema(
        self,
        data: Dict,
        scope: ValidationScope
    ) -> List[Dict]:
        """Validate data against JSON schema."""
        try:
            errors = []
            
            # Get schema for scope
            schema = self.schema_cache.get(scope.value)
            if not schema:
                return [{"error": f"No schema found for scope: {scope.value}"}]
            
            try:
                validate(instance=data, schema=schema)
            except JsonSchemaError as e:
                errors.append({
                    "error": str(e),
                    "path": list(e.path),
                    "type": "schema"
                })
            
            return errors
            
        except Exception as e:
            self.logger.error(f"Schema validation error: {str(e)}")
            return [{"error": str(e), "type": "schema"}]

    async def _validate_rules(
        self,
        data: Dict,
        scope: ValidationScope,
        level: ValidationLevel
    ) -> Tuple[List[Dict], List[Dict]]:
        """Validate data against rules."""
        try:
            errors = []
            warnings = []
            
            # Get rules for scope
            rules = self.validation_rules.get(scope.value, {})
            
            for rule_id, rule in rules.items():
                if rule['level'] != level.value:
                    continue
                
                # Check conditions
                if await self._check_rule_conditions(data, rule['conditions']):
                    # Rule matched, check constraints
                    rule_result = await self._check_rule_constraints(
                        data,
                        rule['constraints']
                    )
                    
                    if not rule_result['valid']:
                        if rule['priority'] == ValidationPriority.HIGH.value:
                            errors.append({
                                "rule_id": rule_id,
                                "error": rule_result['message'],
                                "type": "rule"
                            })
                        else:
                            warnings.append({
                                "rule_id": rule_id,
                                "warning": rule_result['message'],
                                "type": "rule"
                            })
            
            return errors, warnings
            
        except Exception as e:
            self.logger.error(f"Rule validation error: {str(e)}")
            return ([{"error": str(e), "type": "rule"}], [])

    async def _validate_domain(
        self,
        data: Dict,
        scope: ValidationScope,
        level: ValidationLevel
    ) -> Tuple[List[Dict], List[Dict]]:
        """Validate domain-specific requirements."""
        try:
            errors = []
            warnings = []
            
            if scope == ValidationScope.PATTERN:
                # Validate pattern domain rules
                domain_errors = await self._validate_pattern_domain(data)
                errors.extend(domain_errors)
                
            elif scope == ValidationScope.DEPLOYMENT:
                # Validate deployment domain rules
                deploy_errors, deploy_warnings = await self._validate_deployment_domain(
                    data,
                    level
                )
                errors.extend(deploy_errors)
                warnings.extend(deploy_warnings)
                
            elif scope == ValidationScope.SCALING:
                # Validate scaling domain rules
                scaling_errors = await self._validate_scaling_domain(data)
                errors.extend(scaling_errors)
                
            return errors, warnings
            
        except Exception as e:
            self.logger.error(f"Domain validation error: {str(e)}")
            return ([{"error": str(e), "type": "domain"}], [])

    async def _validate_pattern_structure(self, pattern: Dict) -> List[Dict]:
        """Validate pattern structure."""
        try:
            errors = []
            
            required_fields = {
                'name', 'version', 'type', 'parameters', 'rules', 'actions'
            }
            
            # Check required fields
            missing_fields = required_fields - set(pattern.keys())
            if missing_fields:
                errors.append({
                    "error": f"Missing required fields: {missing_fields}",
                    "type": "pattern_structure"
                })
            
            # Validate parameter definitions
            if 'parameters' in pattern:
                param_errors = await self._validate_parameter_definitions(
                    pattern['parameters']
                )
                errors.extend(param_errors)
            
            # Validate rule definitions
            if 'rules' in pattern:
                rule_errors = await self._validate_rule_definitions(
                    pattern['rules']
                )
                errors.extend(rule_errors)
            
            # Validate action definitions
            if 'actions' in pattern:
                action_errors = await self._validate_action_definitions(
                    pattern['actions']
                )
                errors.extend(action_errors)
            
            return errors
            
        except Exception as e:
            self.logger.error(f"Pattern structure validation error: {str(e)}")
            return [{"error": str(e), "type": "pattern_structure"}]

    def _load_validation_rules(self):
        """Load validation rules from configuration."""
        try:
            rules_file = self.config.get('rules_file', 'validation_rules.yaml')
            with open(rules_file, 'r') as f:
                rules = yaml.safe_load(f)
                
            # Process and store rules
            for scope, scope_rules in rules.items():
                self.validation_rules[scope] = {
                    rule['id']: rule
                    for rule in scope_rules
                }
                
        except Exception as e:
            self.logger.error(f"Error loading validation rules: {str(e)}")
            self.validation_rules = {}

    def _load_json_schemas(self):
        """Load JSON schemas from configuration."""
        try:
            schema_dir = self.config.get('schema_dir', 'schemas')
            
            # Load all schema files
            for scope in ValidationScope:
                schema_file = f"{schema_dir}/{scope.value}.json"
                try:
                    with open(schema_file, 'r') as f:
                        self.schema_cache[scope.value] = json.load(f)
                except FileNotFoundError:
                    self.logger.warning(f"Schema file not found: {schema_file}")
                    
        except Exception as e:
            self.logger.error(f"Error loading JSON schemas: {str(e)}")
            self.schema_cache = {}

    async def _store_validation_result(
        self,
        result: ValidationResult,
        data: Dict,
        scope: ValidationScope
    ):
        """Store validation result in history."""
        try:
            # Create history entry
            entry = {
                'timestamp': result.timestamp.isoformat(),
                'scope': scope.value,
                'is_valid': result.is_valid,
                'errors': result.errors,
                'warnings': result.warnings,
                'metadata': result.metadata
            }
            
            # Add to history
            self.validation_history.append(entry)
            
            # Trim history if needed
            if len(self.validation_history) > self.max_history_size:
                self.validation_history = self.validation_history[-self.max_history_size:]
                
        except Exception as e:
            self.logger.error(f"Error storing validation result: {str(e)}")

    async def _check_rule_conditions(
        self,
        data: Dict,
        conditions: List[Dict]
    ) -> bool:
        """Check if data matches rule conditions."""
        try:
            for condition in conditions:
                field = condition['field']
                operator = condition['operator']
                value = condition['value']
                
                # Get field value using dot notation
                field_value = await self._get_field_value(data, field)
                
                # Check condition
                if operator == 'equals' and field_value != value:
                    return False
                elif operator == 'not_equals' and field_value == value:
                    return False
                elif operator == 'contains' and value not in field_value:
                    return False
                elif operator == 'not_contains' and value in field_value:
                    return False
                elif operator == 'greater_than' and field_value <= value:
                    return False
                elif operator == 'less_than' and field_value >= value:
                    return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking rule conditions: {str(e)}")
            return False

    async def _check_rule_constraints(
        self,
        data: Dict,
        constraints: List[Dict]
    ) -> Dict:
        """Check if data satisfies rule constraints."""
        try:
            for constraint in constraints:
                constraint_type = constraint['type']
                
                if constraint_type == 'range':
                    # Check value range
                    value = await self._get_field_value(data, constraint['field'])
                    min_value = constraint.get('min')
                    max_value = constraint.get('max')
                    
                    if min_value is not None and value < min_value:
                        return {
                            'valid': False,
                            'message': f"Value {value} is below minimum {min_value}"
                        }
                        
                    if max_value is not None and value > max_value:
                        return {
                            'valid': False,
                            'message': f"Value {value} is above maximum {max_value}"
                        }
                        
                elif constraint_type == 'pattern':
                    # Check regex pattern
                    value = await self._get_field_value(data, constraint['field'])
                    pattern = constraint['pattern']
                    
                    if not re.match(pattern, str(value)):
                        return {
                            'valid': False,
                            'message': f"Value {value} does not match pattern {pattern}"
                        }
                        
                elif constraint_type == 'dependency':
                    # Check field dependencies
                    field = constraint['field']
                    depends_on = constraint['depends_on']
                    
                    if field in data and depends_on not in data:
                        return {
                            'valid': False,
                            'message': f"Field {field} depends on {depends_on}"
                        }
            
            return {'valid': True, 'message': None}
            
        except Exception as e:
            self.logger.error(f"Error checking rule constraints: {str(e)}")
            return {'valid': False, 'message': str(e)}

    async def _get_field_value(
        self,
        data: Dict,
        field: str
    ) -> Any:
        """Get field value using dot notation."""
        try:
            # Split field path
            parts = field.split('.')
            value = data
            
            # Navigate through path
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                elif isinstance(value, list) and part.isdigit():
                    value = value[int(part)]
                else:
                    raise ValueError(f"Invalid field path: {field}")
                    
                if value is None:
                    break
                    
            return value
            
        except Exception as e:
            self.logger.error(f"Error getting field value: {str(e)}")
            return None