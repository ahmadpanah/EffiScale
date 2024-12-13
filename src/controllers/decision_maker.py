from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
import asyncio
import logging
from enum import Enum
import json
from scipy.stats import norm

class DecisionType(Enum):
    """Types of scaling decisions."""
    SCALE_HORIZONTAL = "scale_horizontal"
    SCALE_VERTICAL = "scale_vertical"
    OPTIMIZE_RESOURCES = "optimize_resources"
    NO_ACTION = "no_action"
    ROLLBACK = "rollback"

class DecisionConfidence(Enum):
    """Confidence levels for decisions."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"

@dataclass
class ScalingDecision:
    """Container for scaling decisions."""
    decision_type: DecisionType
    action: str
    magnitude: float
    confidence: DecisionConfidence
    reason: str
    metadata: Dict
    timestamp: datetime
    impact_assessment: Dict
    recommendations: List[str]

class DecisionMaker:
    """
    Makes intelligent scaling decisions based on multiple inputs including
    metrics, predictions, optimization results, and historical data.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the decision maker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.min_confidence = config.get('decision', {}).get('min_confidence', 0.7)
        self.decision_weights = config.get('weights', {})
        self.performance_targets = config.get('performance', {})
        
        # Decision history
        self.decision_history: List[ScalingDecision] = []
        self.impact_history: List[Tuple[ScalingDecision, Dict]] = []
        
        # State tracking
        self.current_state = {
            'metrics': {},
            'predictions': {},
            'optimization': {},
            'incidents': []
        }
        
        # Initialize impact learning
        self.impact_models = self._initialize_impact_models()
        
        # Start background tasks
        if config.get('auto_learn', {}).get('enabled', True):
            asyncio.create_task(self._impact_learning_loop())

    async def make_decision(
        self,
        current_metrics: Dict[str, float],
        predicted_workload: Optional[List[float]] = None,
        optimization_result: Optional[Dict] = None,
        constraints: Optional[Dict] = None
    ) -> ScalingDecision:
        """
        Make scaling decision based on available information.
        
        Args:
            current_metrics: Current system metrics
            predicted_workload: Optional workload predictions
            optimization_result: Optional resource optimization results
            constraints: Optional operational constraints
            
        Returns:
            Scaling decision
        """
        try:
            # Update state
            self._update_state(current_metrics, predicted_workload, optimization_result)
            
            # Generate decision candidates
            candidates = await self._generate_decision_candidates(constraints)
            
            # Evaluate candidates
            evaluated_candidates = await self._evaluate_candidates(candidates)
            
            # Select best decision
            decision = self._select_best_decision(evaluated_candidates)
            
            # Record decision
            self.decision_history.append(decision)
            
            # Trim history if needed
            max_history = self.config.get('max_history_size', 1000)
            if len(self.decision_history) > max_history:
                self.decision_history = self.decision_history[-max_history:]
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error making decision: {str(e)}")
            raise

    async def _generate_decision_candidates(
        self,
        constraints: Optional[Dict]
    ) -> List[ScalingDecision]:
        """Generate potential decision candidates."""
        candidates = []
        
        # Get current state
        metrics = self.current_state['metrics']
        predictions = self.current_state['predictions']
        optimization = self.current_state['optimization']
        
        # Consider horizontal scaling
        candidates.extend(await self._generate_horizontal_candidates(
            metrics,
            predictions,
            constraints
        ))
        
        # Consider vertical scaling
        candidates.extend(await self._generate_vertical_candidates(
            metrics,
            optimization,
            constraints
        ))
        
        # Consider resource optimization
        candidates.extend(await self._generate_optimization_candidates(
            metrics,
            optimization,
            constraints
        ))
        
        # Always include no-action candidate
        candidates.append(ScalingDecision(
            decision_type=DecisionType.NO_ACTION,
            action="none",
            magnitude=0.0,
            confidence=DecisionConfidence.HIGH,
            reason="No action needed",
            metadata={},
            timestamp=datetime.utcnow(),
            impact_assessment={},
            recommendations=[]
        ))
        
        return candidates

    async def _generate_horizontal_candidates(
        self,
        metrics: Dict[str, float],
        predictions: Dict,
        constraints: Optional[Dict]
    ) -> List[ScalingDecision]:
        """Generate horizontal scaling candidates."""
        candidates = []
        
        # Get current utilization
        cpu_util = metrics.get('cpu_usage', 0)
        memory_util = metrics.get('memory_usage', 0)
        
        # Get predictions if available
        predicted_workload = predictions.get('workload', [])
        
        # Scale out candidate
        if cpu_util > 0.7 or memory_util > 0.7 or \
           (predicted_workload and max(predicted_workload) > 0.7):
            
            candidates.append(ScalingDecision(
                decision_type=DecisionType.SCALE_HORIZONTAL,
                action="scale_out",
                magnitude=self._calculate_scale_out_magnitude(
                    cpu_util,
                    memory_util,
                    predicted_workload
                ),
                confidence=self._calculate_confidence(
                    cpu_util,
                    memory_util,
                    predicted_workload
                ),
                reason="High resource utilization",
                metadata={
                    'cpu_util': cpu_util,
                    'memory_util': memory_util,
                    'predicted_max': max(predicted_workload) if predicted_workload else None
                },
                timestamp=datetime.utcnow(),
                impact_assessment=self._predict_scaling_impact(
                    "scale_out",
                    metrics
                ),
                recommendations=self._generate_recommendations(
                    "scale_out",
                    metrics,
                    predictions
                )
            ))
        
        # Scale in candidate
        if cpu_util < 0.3 and memory_util < 0.3 and \
           (not predicted_workload or max(predicted_workload) < 0.5):
            
            candidates.append(ScalingDecision(
                decision_type=DecisionType.SCALE_HORIZONTAL,
                action="scale_in",
                magnitude=self._calculate_scale_in_magnitude(
                    cpu_util,
                    memory_util,
                    predicted_workload
                ),
                confidence=self._calculate_confidence(
                    cpu_util,
                    memory_util,
                    predicted_workload
                ),
                reason="Low resource utilization",
                metadata={
                    'cpu_util': cpu_util,
                    'memory_util': memory_util,
                    'predicted_max': max(predicted_workload) if predicted_workload else None
                },
                timestamp=datetime.utcnow(),
                impact_assessment=self._predict_scaling_impact(
                    "scale_in",
                    metrics
                ),
                recommendations=self._generate_recommendations(
                    "scale_in",
                    metrics,
                    predictions
                )
            ))
        
        return candidates

    async def _generate_vertical_candidates(
        self,
        metrics: Dict[str, float],
        optimization: Dict,
        constraints: Optional[Dict]
    ) -> List[ScalingDecision]:
        """Generate vertical scaling candidates."""
        candidates = []
        
        if optimization:
            # Get optimization recommendations
            target_cpu = optimization.get('cpu_allocation')
            target_memory = optimization.get('memory_allocation')
            
            if target_cpu is not None and target_memory is not None:
                current_cpu = metrics.get('allocated_cpu', 0)
                current_memory = metrics.get('allocated_memory', 0)
                
                # Scale up candidate
                if target_cpu > current_cpu * 1.1 or target_memory > current_memory * 1.1:
                    candidates.append(ScalingDecision(
                        decision_type=DecisionType.SCALE_VERTICAL,
                        action="scale_up",
                        magnitude=max(
                            target_cpu / current_cpu - 1,
                            target_memory / current_memory - 1
                        ),
                        confidence=DecisionConfidence(optimization.get('confidence', 'MEDIUM')),
                        reason="Resource optimization recommendation",
                        metadata={
                            'target_cpu': target_cpu,
                            'target_memory': target_memory,
                            'current_cpu': current_cpu,
                            'current_memory': current_memory
                        },
                        timestamp=datetime.utcnow(),
                        impact_assessment=self._predict_scaling_impact(
                            "scale_up",
                            metrics,
                            optimization
                        ),
                        recommendations=self._generate_recommendations(
                            "scale_up",
                            metrics,
                            optimization=optimization
                        )
                    ))
                
                # Scale down candidate
                if target_cpu < current_cpu * 0.9 and target_memory < current_memory * 0.9:
                    candidates.append(ScalingDecision(
                        decision_type=DecisionType.SCALE_VERTICAL,
                        action="scale_down",
                        magnitude=max(
                            1 - target_cpu / current_cpu,
                            1 - target_memory / current_memory
                        ),
                        confidence=DecisionConfidence(optimization.get('confidence', 'MEDIUM')),
                        reason="Resource optimization recommendation",
                        metadata={
                            'target_cpu': target_cpu,
                            'target_memory': target_memory,
                            'current_cpu': current_cpu,
                            'current_memory': current_memory
                        },
                        timestamp=datetime.utcnow(),
                        impact_assessment=self._predict_scaling_impact(
                            "scale_down",
                            metrics,
                            optimization
                        ),
                        recommendations=self._generate_recommendations(
                            "scale_down",
                            metrics,
                            optimization=optimization
                        )
                    ))
        
        return candidates

    async def _generate_optimization_candidates(
        self,
        metrics: Dict[str, float],
        optimization: Dict,
        constraints: Optional[Dict]
    ) -> List[ScalingDecision]:
        """Generate resource optimization candidates."""
        candidates = []
        
        if optimization and optimization.get('recommendations'):
            for recommendation in optimization['recommendations']:
                candidates.append(ScalingDecision(
                    decision_type=DecisionType.OPTIMIZE_RESOURCES,
                    action=recommendation['action'],
                    magnitude=recommendation.get('magnitude', 0.0),
                    confidence=DecisionConfidence(recommendation.get('confidence', 'MEDIUM')),
                    reason=recommendation['reason'],
                    metadata=recommendation.get('metadata', {}),
                    timestamp=datetime.utcnow(),
                    impact_assessment=self._predict_scaling_impact(
                        recommendation['action'],
                        metrics,
                        optimization
                    ),
                    recommendations=recommendation.get('recommendations', [])
                ))
        
        return candidates

    async def _evaluate_candidates(
        self,
        candidates: List[ScalingDecision]
    ) -> List[Tuple[ScalingDecision, float]]:
        """Evaluate decision candidates."""
        evaluated = []
        
        for candidate in candidates:
            score = await self._calculate_decision_score(candidate)
            evaluated.append((candidate, score))
        
        return evaluated

    async def _calculate_decision_score(
        self,
        decision: ScalingDecision
    ) -> float:
        """Calculate overall score for a decision."""
        # Base score from confidence
        confidence_scores = {
            DecisionConfidence.HIGH: 1.0,
            DecisionConfidence.MEDIUM: 0.7,
            DecisionConfidence.LOW: 0.4,
            DecisionConfidence.UNCERTAIN: 0.2
        }
        base_score = confidence_scores[decision.confidence]
        
        # Impact score
        impact_score = self._calculate_impact_score(decision.impact_assessment)
        
        # Risk score
        risk_score = self._calculate_risk_score(decision)
        
        # Historical success score
        history_score = await self._calculate_history_score(decision)
        
        # Combine scores using weights
        weights = self.decision_weights
        total_score = (
            weights.get('confidence', 0.3) * base_score +
            weights.get('impact', 0.3) * impact_score +
            weights.get('risk', 0.2) * (1 - risk_score) +  # Invert risk score
            weights.get('history', 0.2) * history_score
        )
        
        return total_score

    def _calculate_impact_score(
        self,
        impact_assessment: Dict
    ) -> float:
        """Calculate score based on predicted impact."""
        if not impact_assessment:
            return 0.5
        
        # Consider performance impact
        perf_impact = impact_assessment.get('performance_impact', 0)
        cost_impact = impact_assessment.get('cost_impact', 0)
        resource_impact = impact_assessment.get('resource_efficiency', 0)
        
        # Normalize and combine impacts
        score = (
            0.4 * (1 + perf_impact) +  # Higher is better
            0.3 * (1 - cost_impact) +   # Lower is better
            0.3 * resource_impact       # Higher is better
        )
        
        return max(0, min(1, score))

    def _calculate_risk_score(
        self,
        decision: ScalingDecision
    ) -> float:
        """Calculate risk score for a decision."""
        base_risk = {
            DecisionType.SCALE_HORIZONTAL: 0.3,
            DecisionType.SCALE_VERTICAL: 0.5,
            DecisionType.OPTIMIZE_RESOURCES: 0.4,
            DecisionType.NO_ACTION: 0.1,
            DecisionType.ROLLBACK: 0.6
        }[decision.decision_type]
        
        # Adjust for magnitude
        magnitude_factor = min(1, decision.magnitude / 2)
        
        # Adjust for confidence
        confidence_factor = {
            DecisionConfidence.HIGH: 0.7,
            DecisionConfidence.MEDIUM: 1.0,
            DecisionConfidence.LOW: 1.3,
            DecisionConfidence.UNCERTAIN: 1.5
        }[decision.confidence]
        
        return min(1, base_risk * magnitude_factor * confidence_factor)

    async def _calculate_history_score(
        self,
        decision: ScalingDecision
    ) -> float:
        """Calculate score based on historical success."""
        similar_decisions = [
            d for d in self.decision_history[-100:]  # Last 100 decisions
            if d.decision_type == decision.decision_type and
            d.action == decision.action
        ]
        
        if not similar_decisions:
            return 0.5
        
        # Calculate success rate from impact history
        success_count = 0
        for d in similar_decisions:
            impacts = [i[1] for i in self.impact_history if i[0] == d]
            if impacts and self._was_decision_successful(impacts[0]):
                success_count += 1
        
        return success_count / len(similar_decisions)

    def _was_decision_successful(
        self,
        impact: Dict
    ) -> bool:
        """Determine if a decision's impact was successful."""
        if not impact:
            return False
        
        # Check performance impact
        perf_success = impact.get('performance_impact', 0) > -0.1
        
        # Check cost impact
        cost_success = impact.get('cost_impact', 0) < 0.2
        
        # Check resource efficiency
        resource_success = impact.get('resource_efficiency', 0) > 0
        
        return perf_success and cost_success and resource_success

    def _select_best_decision(
        self,
        evaluated_candidates: List[Tuple[ScalingDecision, float]]
    ) -> ScalingDecision:
        """Select best decision from candidates."""
        if not evaluated_candidates:
            return ScalingDecision(
                decision_type=DecisionType.NO_ACTION,
                action="none",
                magnitude=0.0,
                confidence=DecisionConfidence.HIGH,
                reason="No candidates available",
                metadata={},
                timestamp=datetime.utcnow(),
                impact_assessment={},
                recommendations=[]
            )
        
        # Sort by score
        sorted_candidates = sorted(
            evaluated_candidates,
            key=lambda x: x[1],
            reverse=True
        )
        
        # Check if best candidate meets minimum confidence
        best_candidate, best_score = sorted_candidates[0]
        
        if best_score < self.min_confidence and \
           best_candidate.decision_type != DecisionType.NO_ACTION:
            return ScalingDecision(
                decision_type=DecisionType.NO_ACTION,
                action="none",
                magnitude=0.0,
                confidence=DecisionConfidence.HIGH,
                reason="No candidate met minimum confidence threshold",
                metadata={'best_score': best_score},
                timestamp=datetime.utcnow(),
                impact_assessment={},
                recommendations=[]
            )
        
        return best_candidate

    def _predict_scaling_impact(
        self,
        action: str,
        metrics: Dict[str, float],
        optimization: Optional[Dict] = None
    ) -> Dict:
        """Predict impact of scaling action."""
        # Get relevant model
        model = self.impact_models.get(action)
        if not model:
            return {}
        
        try:
            # Prepare features
            features = self._prepare_impact_features(
                action,
                metrics,
                optimization
            )
            
            # Predict impacts
            impacts = model.predict([features])[0]
            
            return {
                'performance_impact': float(impacts[0]),
                'cost_impact': float(impacts[1]),
                'resource_efficiency': float(impacts[2])
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting impact: {str(e)}")
            return {}

    def _prepare_impact_features(
        self,
        action: str,
        metrics: Dict[str, float],
        optimization: Optional[Dict]
    ) -> List[float]:
        """Prepare features for impact prediction."""
        features = [
            metrics.get('cpu_usage', 0),
            metrics.get('memory_usage', 0),
            metrics.get('request_rate', 0),
            metrics.get('error_rate', 0)
        ]
        
        if optimization:
            features.extend([
                optimization.get('cpu_allocation', 0),
                optimization.get('memory_allocation', 0)
            ])
        else:
            features.extend([0, 0])
        
        return features

    def _initialize_impact_models(self) -> Dict:
        """Initialize models for impact prediction."""
        # This should be replaced with actual ML models
        return {
            'scale_out': MockImpactModel(),
            'scale_in': MockImpactModel(),
            'scale_up': MockImpactModel(),
            'scale_down': MockImpactModel()
        }

    def _generate_recommendations(
        self,
        action: str,
        metrics: Dict[str, float],
        predictions: Optional[Dict] = None,
        optimization: Optional[Dict] = None
    ) -> List[str]:
        """Generate human-readable recommendations."""
        recommendations = []
        
        if action == "scale_out":
            recommendations.append(
                f"Increase number of instances to handle high utilization "
                f"(CPU: {metrics.get('cpu_usage', 0)*100:.1f}%, "
                f"Memory: {metrics.get('memory_usage', 0)*100:.1f}%)"
            )
            
            if predictions and predictions.get('workload'):
                recommendations.append(
                    f"Prepare for predicted workload increase "
                    f"(Peak: {max(predictions['workload'])*100:.1f}%)"
                )
        
        elif action == "scale_in":
            recommendations.append(
                f"Decrease number of instances to optimize costs "
                f"(Current utilization - CPU: {metrics.get('cpu_usage', 0)*100:.1f}%, "
                f"Memory: {metrics.get('memory_usage', 0)*100:.1f}%)"
            )
        
        elif action in ["scale_up", "scale_down"] and optimization:
            recommendations.append(
                f"Adjust resource allocation based on optimization analysis "
                f"(Target CPU: {optimization.get('cpu_allocation', 0):.1f}, "
                f"Memory: {optimization.get('memory_allocation', 0):.0f}MB)"
            )
        
        return recommendations

    def _update_state(
        self,
        metrics: Dict[str, float],
        predictions: Optional[Dict] = None,
        optimization: Optional[Dict] = None
    ):
        """Update current state."""
        self.current_state['metrics'] = metrics
        if predictions:
            self.current_state['predictions'] = predictions
        if optimization:
            self.current_state['optimization'] = optimization

    async def record_decision_impact(
        self,
        decision: ScalingDecision,
        impact: Dict
    ):
        """Record actual impact of a decision."""
        self.impact_history.append((decision, impact))
        
        # Trim history if needed
        max_history = self.config.get('max_history_size', 1000)
        if len(self.impact_history) > max_history:
            self.impact_history = self.impact_history[-max_history:]

    async def _impact_learning_loop(self):
        """Background task for learning from decision impacts."""
        while True:
            try:
                # Update impact models
                await self._update_impact_models()
                
                # Sleep until next update
                interval = self.config.get('auto_learn', {}).get('interval', 3600)
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in impact learning loop: {str(e)}")
                await asyncio.sleep(60)

    async def _update_impact_models(self):
        """Update impact prediction models with new data."""
        if len(self.impact_history) < self.config.get('min_learning_samples', 50):
            return
        
        # Group impacts by action
        impacts_by_action = {}
        for decision, impact in self.impact_history:
            action = decision.action
            if action not in impacts_by_action:
                impacts_by_action[action] = []
            impacts_by_action[action].append((decision, impact))
        
        # Update models
        for action, impacts in impacts_by_action.items():
            if action in self.impact_models:
                X = []  # Features
                y = []  # Impacts
                
                for decision, impact in impacts:
                    features = self._prepare_impact_features(
                        action,
                        decision.metadata,
                        decision.metadata.get('optimization')
                    )
                    X.append(features)
                    y.append([
                        impact.get('performance_impact', 0),
                        impact.get('cost_impact', 0),
                        impact.get('resource_efficiency', 0)
                    ])
                
                if X and y:
                    self.impact_models[action].fit(X, y)

class MockImpactModel:
    """Mock model for impact prediction."""
    
    def predict(self, X):
        """Mock prediction."""
        return [[0.1, -0.1, 0.2]]
        
    def fit(self, X, y):
        """Mock training."""
        pass

# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'decision': {
            'min_confidence': 0.7
        },
        'weights': {
            'confidence': 0.3,
            'impact': 0.3,
            'risk': 0.2,
            'history': 0.2
        },
        'performance': {
            'target_response_time': 100
        },
        'auto_learn': {
            'enabled': True,
            'interval': 3600
        }
    }
    
    # Initialize decision maker
    decision_maker = DecisionMaker(config)
    
    # Example decision making
    async def main():
        # Current metrics
        metrics = {
            'cpu_usage': 0.8,
            'memory_usage': 0.7,
            'response_time': 150,
            'request_rate': 100,
            'error_rate': 0.01
        }
        
        # Predictions
        predictions = {
            'workload': [0.85, 0.9, 0.82, 0.88]
        }
        
        # Optimization result
        optimization = {
            'cpu_allocation': 2.0,
            'memory_allocation': 4096,
            'confidence': 'HIGH',
            'recommendations': [
                {
                    'action': 'increase_cpu',
                    'magnitude': 0.5,
                    'confidence': 'HIGH',
                    'reason': 'High CPU utilization'
                }
            ]
        }
        
        # Make decision
        decision = await decision_maker.make_decision(
            metrics,
            predictions,
            optimization
        )
        
        print(f"Decision Type: {decision.decision_type.value}")
        print(f"Action: {decision.action}")
        print(f"Magnitude: {decision.magnitude:.2f}")
        print(f"Confidence: {decision.confidence.value}")
        print(f"Reason: {decision.reason}")
        print("Metadata:", json.dumps(decision.metadata, indent=2))
        print("Impact Assessment:", json.dumps(decision.impact_assessment, indent=2))
        print("\nRecommendations:")
        for rec in decision.recommendations:
            print(f"- {rec}")
    
    # Run example
    asyncio.run(main())