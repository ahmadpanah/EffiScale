from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import asyncio
import logging
import json
import numpy as np
from enum import Enum
import pickle
import sklearn.preprocessing as preprocessing
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd
from collections import defaultdict
import aiosqlite
import asyncpg
import hashlib

class KnowledgeType(Enum):
    """Types of knowledge."""
    SCALING_PATTERNS = "scaling_patterns"
    WORKLOAD_PATTERNS = "workload_patterns"
    PERFORMANCE_PATTERNS = "performance_patterns"
    RESOURCE_PATTERNS = "resource_patterns"
    INCIDENT_PATTERNS = "incident_patterns"
    OPTIMIZATION_RULES = "optimization_rules"
    ANOMALY_PATTERNS = "anomaly_patterns"

@dataclass
class KnowledgeEntry:
    """Container for knowledge entries."""
    entry_id: str
    knowledge_type: KnowledgeType
    pattern: Dict
    outcome: Dict
    confidence: float
    metadata: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    
    def to_dict(self) -> Dict:
        """Convert entry to dictionary."""
        return {
            'entry_id': self.entry_id,
            'knowledge_type': self.knowledge_type.value,
            'pattern': self.pattern,
            'outcome': self.outcome,
            'confidence': self.confidence,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'version': self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'KnowledgeEntry':
        """Create entry from dictionary."""
        return cls(
            entry_id=data['entry_id'],
            knowledge_type=KnowledgeType(data['knowledge_type']),
            pattern=data['pattern'],
            outcome=data['outcome'],
            confidence=data['confidence'],
            metadata=data['metadata'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            version=data['version']
        )

class KnowledgeBase:
    """
    Manages system knowledge and learning for autonomous scaling and optimization.
    Implements pattern recognition, learning, and decision support.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the knowledge base.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.knowledge_store: Dict[KnowledgeType, List[KnowledgeEntry]] = defaultdict(list)
        self.pattern_models: Dict[KnowledgeType, Any] = {}
        self.feature_scalers: Dict[KnowledgeType, Any] = {}
        
        # Database connections
        self.postgres_pool = None
        self.sqlite_conn = None
        
        # Knowledge versioning
        self.knowledge_versions: Dict[KnowledgeType, int] = defaultdict(int)
        
        # Pattern matching thresholds
        self.similarity_threshold = config.get('similarity_threshold', 0.8)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        
        # Learning settings
        self.min_samples_for_learning = config.get('min_samples_for_learning', 50)
        self.max_pattern_history = config.get('max_pattern_history', 1000)
        
        # Start background tasks
        asyncio.create_task(self._initialize_storage())
        asyncio.create_task(self._continuous_learning_loop())
        asyncio.create_task(self._knowledge_cleanup_loop())

    async def _initialize_storage(self):
        """Initialize storage connections and load initial knowledge."""
        try:
            # Initialize PostgreSQL if configured
            if 'postgres_url' in self.config:
                self.postgres_pool = await asyncpg.create_pool(
                    self.config['postgres_url'],
                    min_size=5,
                    max_size=10
                )
                await self._initialize_postgres_tables()
            
            # Initialize SQLite if configured
            if 'sqlite_path' in self.config:
                self.sqlite_conn = await aiosqlite.connect(
                    self.config['sqlite_path']
                )
                await self._initialize_sqlite_tables()
            
            # Load initial knowledge
            await self._load_initial_knowledge()
            
            # Initialize models
            self._initialize_models()
            
        except Exception as e:
            self.logger.error(f"Error initializing storage: {str(e)}")
            raise

    async def add_knowledge(
        self,
        knowledge_type: KnowledgeType,
        pattern: Dict,
        outcome: Dict,
        metadata: Optional[Dict] = None
    ) -> KnowledgeEntry:
        """
        Add new knowledge entry.
        
        Args:
            knowledge_type: Type of knowledge
            pattern: Pattern data
            outcome: Outcome data
            metadata: Optional metadata
            
        Returns:
            KnowledgeEntry object
        """
        try:
            # Calculate confidence based on similarity to existing patterns
            confidence = await self._calculate_pattern_confidence(
                knowledge_type,
                pattern,
                outcome
            )
            
            # Create entry
            entry = KnowledgeEntry(
                entry_id=self._generate_entry_id(knowledge_type, pattern),
                knowledge_type=knowledge_type,
                pattern=pattern,
                outcome=outcome,
                confidence=confidence,
                metadata=metadata or {},
                version=self.knowledge_versions[knowledge_type] + 1
            )
            
            # Store entry
            await self._store_knowledge_entry(entry)
            
            # Update version
            self.knowledge_versions[knowledge_type] += 1
            
            # Trigger learning if enough samples
            if len(self.knowledge_store[knowledge_type]) >= self.min_samples_for_learning:
                asyncio.create_task(self._update_models(knowledge_type))
            
            return entry
            
        except Exception as e:
            self.logger.error(f"Error adding knowledge: {str(e)}")
            raise

    async def find_matching_patterns(
        self,
        knowledge_type: KnowledgeType,
        pattern: Dict,
        min_confidence: Optional[float] = None
    ) -> List[KnowledgeEntry]:
        """Find matching patterns for given input."""
        try:
            matches = []
            min_conf = min_confidence or self.confidence_threshold
            
            # Get model for pattern type
            model = self.pattern_models.get(knowledge_type)
            scaler = self.feature_scalers.get(knowledge_type)
            
            if model and scaler:
                # Prepare input features
                features = self._prepare_pattern_features(pattern)
                scaled_features = scaler.transform([features])
                
                # Get predictions and probabilities
                if isinstance(model, RandomForestClassifier):
                    probs = model.predict_proba(scaled_features)[0]
                    pred_idx = np.argmax(probs)
                    confidence = probs[pred_idx]
                else:
                    pred = model.predict(scaled_features)[0]
                    confidence = self._calculate_regression_confidence(
                        model,
                        scaled_features,
                        pred
                    )
                
                if confidence >= min_conf:
                    # Find similar patterns in knowledge store
                    similar_patterns = self._find_similar_patterns(
                        knowledge_type,
                        pattern,
                        self.similarity_threshold
                    )
                    matches.extend(similar_patterns)
            
            return matches
            
        except Exception as e:
            self.logger.error(f"Error finding patterns: {str(e)}")
            return []

    async def get_knowledge(
        self,
        knowledge_type: KnowledgeType,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        min_confidence: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[KnowledgeEntry]:
        """Get knowledge entries based on criteria."""
        try:
            entries = []
            min_conf = min_confidence or self.confidence_threshold
            
            if self.postgres_pool:
                # Query PostgreSQL
                query = """
                    SELECT * FROM knowledge_entries
                    WHERE knowledge_type = $1
                    AND confidence >= $2
                """
                params = [knowledge_type.value, min_conf]
                
                if start_time:
                    query += " AND timestamp >= $3"
                    params.append(start_time)
                
                if end_time:
                    query += " AND timestamp <= $4"
                    params.append(end_time)
                
                query += " ORDER BY timestamp DESC"
                
                if limit:
                    query += f" LIMIT {limit}"
                
                async with self.postgres_pool.acquire() as conn:
                    rows = await conn.fetch(query, *params)
                    
                    for row in rows:
                        entries.append(KnowledgeEntry(
                            entry_id=row['entry_id'],
                            knowledge_type=KnowledgeType(row['knowledge_type']),
                            pattern=json.loads(row['pattern']),
                            outcome=json.loads(row['outcome']),
                            confidence=row['confidence'],
                            metadata=json.loads(row['metadata']),
                            timestamp=row['timestamp'],
                            version=row['version']
                        ))
            
            return entries
            
        except Exception as e:
            self.logger.error(f"Error getting knowledge: {str(e)}")
            return []

    async def update_confidence(
        self,
        entry_id: str,
        new_confidence: float
    ) -> bool:
        """Update confidence for a knowledge entry."""
        try:
            # Update in memory
            for entries in self.knowledge_store.values():
                for entry in entries:
                    if entry.entry_id == entry_id:
                        entry.confidence = new_confidence
                        break
            
            # Update in PostgreSQL if available
            if self.postgres_pool:
                async with self.postgres_pool.acquire() as conn:
                    await conn.execute("""
                        UPDATE knowledge_entries
                        SET confidence = $1
                        WHERE entry_id = $2
                    """, new_confidence, entry_id)
            
            # Update in SQLite if available
            if self.sqlite_conn:
                await self.sqlite_conn.execute("""
                    UPDATE knowledge_entries
                    SET confidence = ?
                    WHERE entry_id = ?
                """, (new_confidence, entry_id))
                await self.sqlite_conn.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating confidence: {str(e)}")
            return False

    async def _store_knowledge_entry(
        self,
        entry: KnowledgeEntry
    ):
        """Store knowledge entry."""
        try:
            # Store in memory
            self.knowledge_store[entry.knowledge_type].append(entry)
            
            # Trim if needed
            if len(self.knowledge_store[entry.knowledge_type]) > self.max_pattern_history:
                self.knowledge_store[entry.knowledge_type] = \
                    self.knowledge_store[entry.knowledge_type][-self.max_pattern_history:]
            
            # Store in PostgreSQL if available
            if self.postgres_pool:
                async with self.postgres_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO knowledge_entries (
                            entry_id, knowledge_type, pattern, outcome,
                            confidence, metadata, timestamp, version
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    entry.entry_id,
                    entry.knowledge_type.value,
                    json.dumps(entry.pattern),
                    json.dumps(entry.outcome),
                    entry.confidence,
                    json.dumps(entry.metadata),
                    entry.timestamp,
                    entry.version
                    )
            
            # Store in SQLite if available
            if self.sqlite_conn:
                await self.sqlite_conn.execute("""
                    INSERT INTO knowledge_entries (
                        entry_id, knowledge_type, pattern, outcome,
                        confidence, metadata, timestamp, version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.entry_id,
                    entry.knowledge_type.value,
                    json.dumps(entry.pattern),
                    json.dumps(entry.outcome),
                    entry.confidence,
                    json.dumps(entry.metadata),
                    entry.timestamp.isoformat(),
                    entry.version
                ))
                await self.sqlite_conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error storing knowledge entry: {str(e)}")
            raise

    def _initialize_models(self):
        """Initialize pattern recognition models."""
        for knowledge_type in KnowledgeType:
            # Initialize feature scaler
            self.feature_scalers[knowledge_type] = preprocessing.StandardScaler()
            
            # Initialize model based on knowledge type
            if knowledge_type in [KnowledgeType.SCALING_PATTERNS, 
                                KnowledgeType.ANOMALY_PATTERNS]:
                self.pattern_models[knowledge_type] = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
            else:
                self.pattern_models[knowledge_type] = RandomForestRegressor(
                    n_estimators=100,
                    random_state=42
                )

    async def _update_models(
        self,
        knowledge_type: KnowledgeType
    ):
        """Update pattern recognition models."""
        try:
            entries = self.knowledge_store[knowledge_type]
            
            if len(entries) < self.min_samples_for_learning:
                return
            
            # Prepare training data
            X = []  # Features
            y = []  # Outcomes
            
            for entry in entries:
                features = self._prepare_pattern_features(entry.pattern)
                outcome = self._prepare_outcome_target(entry.outcome)
                
                X.append(features)
                y.append(outcome)
            
            X = np.array(X)
            y = np.array(y)
            
            # Update scaler
            self.feature_scalers[knowledge_type].fit(X)
            X_scaled = self.feature_scalers[knowledge_type].transform(X)
            
            # Update model
            model = self.pattern_models[knowledge_type]
            model.fit(X_scaled, y)
            
        except Exception as e:
            self.logger.error(f"Error updating models: {str(e)}")

    def _prepare_pattern_features(
        self,
        pattern: Dict
    ) -> List[float]:
        """Prepare pattern features for model input."""
        # This should be customized based on your pattern structure
        features = []
        
        # Extract numeric features
        for key in ['cpu_usage', 'memory_usage', 'request_rate', 'error_rate']:
            features.append(float(pattern.get(key, 0)))
        
        return features

    def _prepare_outcome_target(
        self,
        outcome: Dict
    ) -> Union[float, int]:
        """Prepare outcome target for model training."""
        # This should be customized based on your outcome structure
        return float(outcome.get('value', 0))

    def _calculate_regression_confidence(
        self,
        model: RandomForestRegressor,
        features: np.ndarray,
        prediction: float
    ) -> float:
        """Calculate confidence for regression prediction."""
        # Use prediction std from forest as confidence measure
        predictions = []
        for estimator in model.estimators_:
            predictions.append(estimator.predict(features)[0])
        
        std = np.std(predictions)
        confidence = 1 / (1 + std)
        
        return confidence

    def _find_similar_patterns(
        self,
        knowledge_type: KnowledgeType,
        pattern: Dict,
        threshold: float
    ) -> List[KnowledgeEntry]:
        """Find patterns similar to input pattern."""
        similar = []
        pattern_features = np.array(self._prepare_pattern_features(pattern))
        
        for entry in self.knowledge_store[knowledge_type]:
            entry_features = np.array(self._prepare_pattern_features(entry.pattern))
            
            # Calculate cosine similarity
            similarity = np.dot(pattern_features, entry_features) / \
                        (np.linalg.norm(pattern_features) * np.linalg.norm(entry_features))
            
            if similarity >= threshold:
                similar.append(entry)
        
        return similar

    async def _calculate_pattern_confidence(
        self,
        knowledge_type: KnowledgeType,
        pattern: Dict,
        outcome: Dict
    ) -> float:
        """Calculate confidence for new pattern."""
        try:
            # Find similar patterns
            similar_patterns = self._find_similar_patterns(
                knowledge_type,
                pattern,
                self.similarity_threshold
            )
            
            if not similar_patterns:
                return 0.5  # Base confidence for new patterns
            
            # Calculate confidence based on outcome similarity
            confidences = []
            for entry in similar_patterns:
                outcome_similarity = self._calculate_outcome_similarity(
                    outcome,
                    entry.outcome
                )
                confidences.append(outcome_similarity * entry.confidence)
            
            return np.mean(confidences)
            
        except Exception:
            return 0.5

    def _calculate_outcome_similarity(
        self,
        outcome1: Dict,
        outcome2: Dict
    ) -> float:
        """Calculate similarity between outcomes."""
        try:
            # This should be customized based on your outcome structure
            value1 = float(outcome1.get('value', 0))
            value2 = float(outcome2.get('value', 0))
            
            # Calculate similarity based on relative difference
            max_val = max(abs(value1), abs(value2))
            if max_val == 0:
                return 1.0
            
            diff = abs(value1 - value2) / max_val
            similarity = 1 - min(diff, 1.0)
            
            return similarity
            
        except Exception:
            return 0.0

    def _generate_entry_id(
        self,
        knowledge_type: KnowledgeType,
        pattern: Dict
    ) -> str:
        """Generate unique entry ID."""
        data = f"{knowledge_type.value}:{json.dumps(pattern, sort_keys=True)}:{datetime.utcnow().isoformat()}"
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    async def _continuous_learning_loop(self):
        """Background task for continuous learning."""
        while True:
            try:
                for knowledge_type in KnowledgeType:
                    if len(self.knowledge_store[knowledge_type]) >= self.min_samples_for_learning:
                        await self._update_models(knowledge_type)
                
                # Sleep before next learning cycle
                await asyncio.sleep(
                    self.config.get('learning_interval', 3600)
                )
                
            except Exception as e:
                self.logger.error(f"Error in learning loop: {str(e)}")
                await asyncio.sleep(60)

    async def _knowledge_cleanup_loop(self):
        """Background task for cleaning up old knowledge entries."""
        while True:
            try:
                # Get retention period
                retention_days = self.config.get('knowledge_retention_days', 90)
                cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
                
                # Clean up PostgreSQL
                if self.postgres_pool:
                    async with self.postgres_pool.acquire() as conn:
                        await conn.execute("""
                            DELETE FROM knowledge_entries
                            WHERE timestamp < $1
                        """, cutoff_date)
                
                # Clean up SQLite
                if self.sqlite_conn:
                    await self.sqlite_conn.execute("""
                        DELETE FROM knowledge_entries
                        WHERE timestamp < ?
                    """, (cutoff_date.isoformat(),))
                    await self.sqlite_conn.commit()
                
                # Clean up memory
                for knowledge_type in KnowledgeType:
                    self.knowledge_store[knowledge_type] = [
                        entry for entry in self.knowledge_store[knowledge_type]
                        if entry.timestamp > cutoff_date
                    ]
                
                # Sleep before next cleanup
                await asyncio.sleep(
                    self.config.get('cleanup_interval', 86400)  # 24 hours
                )
                
            except Exception as e:
                self.logger.error(f"Error in knowledge cleanup: {str(e)}")
                await asyncio.sleep(3600)

    async def _initialize_postgres_tables(self):
        """Initialize PostgreSQL tables."""
        async with self.postgres_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_entries (
                    entry_id TEXT PRIMARY KEY,
                    knowledge_type TEXT NOT NULL,
                    pattern JSONB NOT NULL,
                    outcome JSONB NOT NULL,
                    confidence FLOAT NOT NULL,
                    metadata JSONB,
                    timestamp TIMESTAMP NOT NULL,
                    version INTEGER NOT NULL
                );
                
                CREATE INDEX IF NOT EXISTS idx_knowledge_entries_type_time 
                ON knowledge_entries (knowledge_type, timestamp);
                
                CREATE INDEX IF NOT EXISTS idx_knowledge_entries_confidence 
                ON knowledge_entries (confidence);
            """)

    async def _initialize_sqlite_tables(self):
        """Initialize SQLite tables."""
        await self.sqlite_conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_entries (
                entry_id TEXT PRIMARY KEY,
                knowledge_type TEXT NOT NULL,
                pattern TEXT NOT NULL,
                outcome TEXT NOT NULL,
                confidence REAL NOT NULL,
                metadata TEXT,
                timestamp TEXT NOT NULL,
                version INTEGER NOT NULL
            );
            
            CREATE INDEX IF NOT EXISTS idx_knowledge_entries_type_time 
            ON knowledge_entries (knowledge_type, timestamp);
            
            CREATE INDEX IF NOT EXISTS idx_knowledge_entries_confidence 
            ON knowledge_entries (confidence);
        """)
        await self.sqlite_conn.commit()

    async def _load_initial_knowledge(self):
        """Load initial knowledge from storage."""
        try:
            # Try to load from PostgreSQL first
            if self.postgres_pool:
                async with self.postgres_pool.acquire() as conn:
                    rows = await conn.fetch("""
                        SELECT * FROM knowledge_entries
                        ORDER BY timestamp DESC
                    """)
                    
                    for row in rows:
                        entry = KnowledgeEntry(
                            entry_id=row['entry_id'],
                            knowledge_type=KnowledgeType(row['knowledge_type']),
                            pattern=json.loads(row['pattern']),
                            outcome=json.loads(row['outcome']),
                            confidence=row['confidence'],
                            metadata=json.loads(row['metadata']),
                            timestamp=row['timestamp'],
                            version=row['version']
                        )
                        
                        self.knowledge_store[entry.knowledge_type].append(entry)
                        self.knowledge_versions[entry.knowledge_type] = max(
                            self.knowledge_versions[entry.knowledge_type],
                            entry.version
                        )
            
            # Fall back to SQLite if needed
            elif self.sqlite_conn:
                async with self.sqlite_conn.execute("""
                    SELECT * FROM knowledge_entries
                    ORDER BY timestamp DESC
                """) as cursor:
                    async for row in cursor:
                        entry = KnowledgeEntry(
                            entry_id=row[0],
                            knowledge_type=KnowledgeType(row[1]),
                            pattern=json.loads(row[2]),
                            outcome=json.loads(row[3]),
                            confidence=row[4],
                            metadata=json.loads(row[5]),
                            timestamp=datetime.fromisoformat(row[6]),
                            version=row[7]
                        )
                        
                        self.knowledge_store[entry.knowledge_type].append(entry)
                        self.knowledge_versions[entry.knowledge_type] = max(
                            self.knowledge_versions[entry.knowledge_type],
                            entry.version
                        )
            
        except Exception as e:
            self.logger.error(f"Error loading initial knowledge: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'postgres_url': 'postgresql://user:password@localhost/dbname',
        'sqlite_path': 'knowledge.db',
        'similarity_threshold': 0.8,
        'confidence_threshold': 0.7,
        'min_samples_for_learning': 50,
        'max_pattern_history': 1000,
        'learning_interval': 3600,
        'knowledge_retention_days': 90
    }
    
    # Initialize knowledge base
    kb = KnowledgeBase(config)
    
    # Example knowledge management
    async def main():
        # Add new knowledge
        pattern = {
            'cpu_usage': 0.8,
            'memory_usage': 0.7,
            'request_rate': 100,
            'error_rate': 0.01
        }
        
        outcome = {
            'action': 'scale_out',
            'magnitude': 2,
            'success': True,
            'value': 1.0
        }
        
        entry = await kb.add_knowledge(
            KnowledgeType.SCALING_PATTERNS,
            pattern,
            outcome,
            {'source': 'auto_scaler'}
        )
        
        print(f"Added knowledge entry: {entry.entry_id}")
        
        # Find matching patterns
        new_pattern = {
            'cpu_usage': 0.75,
            'memory_usage': 0.65,
            'request_rate': 90,
            'error_rate': 0.02
        }
        
        matches = await kb.find_matching_patterns(
            KnowledgeType.SCALING_PATTERNS,
            new_pattern,
            min_confidence=0.6
        )
        
        print(f"Found {len(matches)} matching patterns")
        
        for match in matches:
            print(f"Match {match.entry_id} with confidence {match.confidence:.2f}")
            print("Pattern:", json.dumps(match.pattern, indent=2))
            print("Outcome:", json.dumps(match.outcome, indent=2))
    
    # Run example
    asyncio.run(main())