from typing import Dict, List, Optional, Tuple, Union, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import asyncio
import logging
import json
import numpy as np
from enum import Enum
import hashlib
from collections import defaultdict
import aiosqlite
import asyncpg
import pandas as pd
from scipy import signal, stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

class PatternType(Enum):
    """Types of system patterns."""
    WORKLOAD = "workload"
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    ANOMALY = "anomaly"
    SCALING = "scaling"
    INCIDENT = "incident"

class PatternCategory(Enum):
    """Categories of patterns."""
    SPIKE = "spike"
    GRADUAL = "gradual"
    PERIODIC = "periodic"
    RANDOM = "random"
    SEASONAL = "seasonal"
    TREND = "trend"
    COMPOUND = "compound"

@dataclass
class Pattern:
    """Container for system patterns."""
    pattern_id: str
    pattern_type: PatternType
    category: PatternCategory
    data: Dict
    features: Dict
    metadata: Dict
    timestamp: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 1.0
    frequency: int = 1
    tags: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict:
        """Convert pattern to dictionary."""
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type.value,
            'category': self.category.value,
            'data': self.data,
            'features': self.features,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence,
            'frequency': self.frequency,
            'tags': list(self.tags)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Pattern':
        """Create pattern from dictionary."""
        return cls(
            pattern_id=data['pattern_id'],
            pattern_type=PatternType(data['pattern_type']),
            category=PatternCategory(data['category']),
            data=data['data'],
            features=data['features'],
            metadata=data['metadata'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            confidence=data['confidence'],
            frequency=data['frequency'],
            tags=set(data['tags'])
        )

class PatternLibrary:
    """
    Manages a library of system behavior patterns.
    Implements pattern recognition, storage, and analysis.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the pattern library.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.patterns: Dict[PatternType, List[Pattern]] = defaultdict(list)
        self.pattern_index: Dict[str, Pattern] = {}
        
        # Feature extraction settings
        self.window_size = config.get('window_size', 60)  # 60 data points
        self.min_pattern_length = config.get('min_pattern_length', 10)
        
        # Pattern matching settings
        self.similarity_threshold = config.get('similarity_threshold', 0.8)
        self.clustering_eps = config.get('clustering_eps', 0.3)
        self.clustering_min_samples = config.get('clustering_min_samples', 5)
        
        # Database connections
        self.postgres_pool = None
        self.sqlite_conn = None
        
        # Initialize preprocessing
        self.scaler = StandardScaler()
        
        # Pattern statistics
        self.pattern_stats: Dict[PatternType, Dict] = defaultdict(dict)
        
        # Start background tasks
        asyncio.create_task(self._initialize_storage())
        asyncio.create_task(self._pattern_analysis_loop())
        asyncio.create_task(self._pattern_cleanup_loop())

    async def add_pattern(
        self,
        pattern_type: PatternType,
        data: Dict,
        metadata: Optional[Dict] = None,
        tags: Optional[Set[str]] = None
    ) -> Optional[Pattern]:
        """
        Add new pattern to library.
        
        Args:
            pattern_type: Type of pattern
            data: Pattern data
            metadata: Optional metadata
            tags: Optional tags
            
        Returns:
            Pattern object if added successfully
        """
        try:
            # Extract features
            features = self._extract_features(data)
            if not features:
                return None
            
            # Determine pattern category
            category = self._categorize_pattern(features)
            
            # Generate pattern ID
            pattern_id = self._generate_pattern_id(pattern_type, features)
            
            # Check for existing similar patterns
            similar_pattern = await self.find_similar_pattern(
                pattern_type,
                features
            )
            
            if similar_pattern:
                # Update existing pattern
                similar_pattern.frequency += 1
                similar_pattern.confidence = min(1.0, similar_pattern.confidence + 0.1)
                if tags:
                    similar_pattern.tags.update(tags)
                return similar_pattern
            
            # Create new pattern
            pattern = Pattern(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                category=category,
                data=data,
                features=features,
                metadata=metadata or {},
                tags=tags or set()
            )
            
            # Store pattern
            await self._store_pattern(pattern)
            
            return pattern
            
        except Exception as e:
            self.logger.error(f"Error adding pattern: {str(e)}")
            return None

    def _extract_features(self, data: Dict) -> Dict:
        """Extract features from pattern data."""
        try:
            features = {}
            
            # Time series features
            if 'values' in data:
                values = np.array(data['values'])
                if len(values) < self.min_pattern_length:
                    return {}
                
                # Basic statistics
                features.update({
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'range': np.ptp(values),
                    'skewness': stats.skew(values),
                    'kurtosis': stats.kurtosis(values)
                })
                
                # Trend features
                trend = signal.detrend(values)
                features['trend_strength'] = 1 - (np.var(trend) / np.var(values))
                
                # Periodicity features
                if len(values) > 10:
                    f, Pxx = signal.periodogram(values)
                    if len(Pxx) > 1:
                        features['dominant_frequency'] = f[np.argmax(Pxx[1:])]
                        features['spectral_entropy'] = stats.entropy(Pxx)
                
                # Rate of change features
                diff = np.diff(values)
                features.update({
                    'mean_rate_of_change': np.mean(diff),
                    'max_rate_of_change': np.max(np.abs(diff))
                })
            
            # Categorical features
            for key in ['pattern_type', 'source', 'region']:
                if key in data:
                    features[f'has_{key}'] = 1
                    features[f'{key}_value'] = hash(str(data[key])) % 1000
            
            # Normalize features
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    features[key] = float(value)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            return {}

    def _categorize_pattern(
        self,
        features: Dict
    ) -> PatternCategory:
        """Determine pattern category from features."""
        try:
            # Check for periodic patterns
            if 'dominant_frequency' in features and \
               features['spectral_entropy'] < 0.7:
                return PatternCategory.PERIODIC
            
            # Check for spikes
            if 'max_rate_of_change' in features and \
               features['max_rate_of_change'] > 3 * features.get('std', 0):
                return PatternCategory.SPIKE
            
            # Check for gradual changes
            if 'trend_strength' in features and \
               features['trend_strength'] > 0.7:
                return PatternCategory.GRADUAL
            
            # Check for seasonal patterns
            if 'values' in features and \
               self._check_seasonality(features['values']):
                return PatternCategory.SEASONAL
            
            # Check for trends
            if 'trend_strength' in features and \
               features['trend_strength'] > 0.3:
                return PatternCategory.TREND
            
            # Default to random if no specific pattern is found
            return PatternCategory.RANDOM
            
        except Exception:
            return PatternCategory.RANDOM

    def _check_seasonality(
        self,
        values: List[float],
        threshold: float = 0.7
    ) -> bool:
        """Check if pattern shows seasonality."""
        try:
            if len(values) < 2 * self.window_size:
                return False
            
            # Calculate autocorrelation
            acf = pd.Series(values).autocorr(lag=self.window_size)
            
            return abs(acf) > threshold
            
        except Exception:
            return False

    async def find_similar_pattern(
        self,
        pattern_type: PatternType,
        features: Dict
    ) -> Optional[Pattern]:
        """Find similar existing pattern."""
        try:
            if not self.patterns[pattern_type]:
                return None
            
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features)
            
            # Get feature vectors for existing patterns
            existing_vectors = [
                self._prepare_feature_vector(p.features)
                for p in self.patterns[pattern_type]
            ]
            
            if not existing_vectors:
                return None
            
            # Calculate similarities
            similarities = [
                self._calculate_similarity(feature_vector, ev)
                for ev in existing_vectors
            ]
            
            # Find most similar pattern
            max_similarity = max(similarities)
            if max_similarity >= self.similarity_threshold:
                max_index = similarities.index(max_similarity)
                return self.patterns[pattern_type][max_index]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding similar pattern: {str(e)}")
            return None

    def _prepare_feature_vector(
        self,
        features: Dict
    ) -> np.ndarray:
        """Prepare feature vector for similarity calculation."""
        # Use common features in consistent order
        common_features = [
            'mean', 'std', 'min', 'max', 'range',
            'skewness', 'kurtosis', 'trend_strength'
        ]
        
        vector = [
            features.get(feature, 0.0)
            for feature in common_features
        ]
        
        return np.array(vector)

    def _calculate_similarity(
        self,
        vector1: np.ndarray,
        vector2: np.ndarray
    ) -> float:
        """Calculate similarity between feature vectors."""
        try:
            # Normalize vectors
            norm1 = np.linalg.norm(vector1)
            norm2 = np.linalg.norm(vector2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            vector1_norm = vector1 / norm1
            vector2_norm = vector2 / norm2
            
            # Calculate cosine similarity
            similarity = np.dot(vector1_norm, vector2_norm)
            
            return float(similarity)
            
        except Exception:
            return 0.0

    async def _store_pattern(
        self,
        pattern: Pattern
    ):
        """Store pattern in library."""
        try:
            # Store in memory
            self.patterns[pattern.pattern_type].append(pattern)
            self.pattern_index[pattern.pattern_id] = pattern
            
            # Store in PostgreSQL if available
            if self.postgres_pool:
                async with self.postgres_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO patterns (
                            pattern_id, pattern_type, category, data,
                            features, metadata, timestamp, confidence,
                            frequency, tags
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        ON CONFLICT (pattern_id) DO UPDATE
                        SET frequency = patterns.frequency + 1,
                            confidence = $8,
                            tags = $10
                    """,
                    pattern.pattern_id,
                    pattern.pattern_type.value,
                    pattern.category.value,
                    json.dumps(pattern.data),
                    json.dumps(pattern.features),
                    json.dumps(pattern.metadata),
                    pattern.timestamp,
                    pattern.confidence,
                    pattern.frequency,
                    list(pattern.tags)
                    )
            
            # Store in SQLite if available
            if self.sqlite_conn:
                await self.sqlite_conn.execute("""
                    INSERT OR REPLACE INTO patterns (
                        pattern_id, pattern_type, category, data,
                        features, metadata, timestamp, confidence,
                        frequency, tags
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pattern.pattern_id,
                    pattern.pattern_type.value,
                    pattern.category.value,
                    json.dumps(pattern.data),
                    json.dumps(pattern.features),
                    json.dumps(pattern.metadata),
                    pattern.timestamp.isoformat(),
                    pattern.confidence,
                    pattern.frequency,
                    json.dumps(list(pattern.tags))
                ))
                await self.sqlite_conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error storing pattern: {str(e)}")
            raise

    def _generate_pattern_id(
        self,
        pattern_type: PatternType,
        features: Dict
    ) -> str:
        """Generate unique pattern ID."""
        data = f"{pattern_type.value}:{json.dumps(features, sort_keys=True)}"
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    async def get_patterns(
        self,
        pattern_type: Optional[PatternType] = None,
        category: Optional[PatternCategory] = None,
        tags: Optional[Set[str]] = None,
        min_confidence: float = 0.0,
        limit: Optional[int] = None
    ) -> List[Pattern]:
        """Get patterns based on criteria."""
        try:
            patterns = []
            
            if self.postgres_pool:
                # Build query
                query = "SELECT * FROM patterns WHERE confidence >= $1"
                params = [min_confidence]
                
                if pattern_type:
                    query += " AND pattern_type = $2"
                    params.append(pattern_type.value)
                
                if category:
                    query += " AND category = $3"
                    params.append(category.value)
                
                if tags:
                    query += " AND tags @> $4"
                    params.append(list(tags))
                
                query += " ORDER BY frequency DESC"
                
                if limit:
                    query += f" LIMIT {limit}"
                
                # Execute query
                async with self.postgres_pool.acquire() as conn:
                    rows = await conn.fetch(query, *params)
                    
                    for row in rows:
                        patterns.append(Pattern(
                            pattern_id=row['pattern_id'],
                            pattern_type=PatternType(row['pattern_type']),
                            category=PatternCategory(row['category']),
                            data=json.loads(row['data']),
                            features=json.loads(row['features']),
                            metadata=json.loads(row['metadata']),
                            timestamp=row['timestamp'],
                            confidence=row['confidence'],
                            frequency=row['frequency'],
                            tags=set(row['tags'])
                        ))
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error getting patterns: {str(e)}")
            return []

    async def analyze_patterns(
        self,
        pattern_type: PatternType
    ) -> Dict:
        """Analyze patterns of given type."""
        try:
            patterns = self.patterns[pattern_type]
            if not patterns:
                return {}
            
            # Prepare feature vectors
            vectors = []
            for pattern in patterns:
                vector = self._prepare_feature_vector(pattern.features)
                vectors.append(vector)
            
            vectors = np.array(vectors)
            
            # Scale features
            vectors_scaled = self.scaler.fit_transform(vectors)
            
            # Cluster patterns
            clustering = DBSCAN(
                eps=self.clustering_eps,
                min_samples=self.clustering_min_samples
            ).fit(vectors_scaled)
            
            # Analyze clusters
            n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
            
            analysis = {
                'pattern_count': len(patterns),
                'cluster_count': n_clusters,
                'categories': defaultdict(int),
                'cluster_sizes': defaultdict(int),
                'frequent_tags': defaultdict(int),
                'temporal_distribution': self._analyze_temporal_distribution(patterns)
            }
            
            # Count categories and tags
            for pattern in patterns:
                analysis['categories'][pattern.category.value] += 1
                for tag in pattern.tags:
                    analysis['frequent_tags'][tag] += 1
            
            # Count cluster sizes
            for label in clustering.labels_:
                if label != -1:
                    analysis['cluster_sizes'][f'cluster_{label}'] += 1
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing patterns: {str(e)}")
            return {}

    def _analyze_temporal_distribution(
        self,
        patterns: List[Pattern]
    ) -> Dict:
        """Analyze temporal distribution of patterns."""
        try:
            timestamps = [p.timestamp for p in patterns]
            if not timestamps:
                return {}
            
            # Calculate time ranges
            min_time = min(timestamps)
            max_time = max(timestamps)
            time_range = (max_time - min_time).total_seconds()
            
            # Create time bins
            n_bins = min(50, len(timestamps))
            bins = np.linspace(min_time.timestamp(), max_time.timestamp(), n_bins)
            
            # Count patterns in each bin
            hist, _ = np.histogram([t.timestamp() for t in timestamps], bins=bins)
            
            return {
                'min_time': min_time.isoformat(),
                'max_time': max_time.isoformat(),
                'time_range_seconds': time_range,
                'distribution': hist.tolist(),
                'bin_edges': [datetime.fromtimestamp(b).isoformat() for b in bins]
            }
            
        except Exception:
            return {}

    async def _pattern_analysis_loop(self):
        """Background task for pattern analysis."""
        while True:
            try:
                # Analyze patterns for each type
                for pattern_type in PatternType:
                    analysis = await self.analyze_patterns(pattern_type)
                    self.pattern_stats[pattern_type] = analysis
                
                # Sleep before next analysis
                await asyncio.sleep(
                    self.config.get('analysis_interval', 3600)  # 1 hour
                )
                
            except Exception as e:
                self.logger.error(f"Error in pattern analysis: {str(e)}")
                await asyncio.sleep(60)

    async def _pattern_cleanup_loop(self):
        """Background task for cleaning up old patterns."""
        while True:
            try:
                # Get retention period
                retention_days = self.config.get('pattern_retention_days', 30)
                cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
                
                # Clean up PostgreSQL
                if self.postgres_pool:
                    async with self.postgres_pool.acquire() as conn:
                        await conn.execute("""
                            DELETE FROM patterns
                            WHERE timestamp < $1
                            AND frequency < $2
                        """, cutoff_date, self.config.get('min_frequency', 2))
                
                # Clean up SQLite
                if self.sqlite_conn:
                    await self.sqlite_conn.execute("""
                        DELETE FROM patterns
                        WHERE timestamp < ?
                        AND frequency < ?
                    """, (cutoff_date.isoformat(), self.config.get('min_frequency', 2)))
                    await self.sqlite_conn.commit()
                
                # Clean up memory
                for pattern_type in PatternType:
                    self.patterns[pattern_type] = [
                        p for p in self.patterns[pattern_type]
                        if p.timestamp > cutoff_date or p.frequency >= self.config.get('min_frequency', 2)
                    ]
                
                # Sleep before next cleanup
                await asyncio.sleep(
                    self.config.get('cleanup_interval', 86400)  # 24 hours
                )
                
            except Exception as e:
                self.logger.error(f"Error in pattern cleanup: {str(e)}")
                await asyncio.sleep(3600)

    async def _initialize_storage(self):
        """Initialize storage connections."""
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
            
        except Exception as e:
            self.logger.error(f"Error initializing storage: {str(e)}")
            raise

    async def _initialize_postgres_tables(self):
        """Initialize PostgreSQL tables."""
        async with self.postgres_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    category TEXT NOT NULL,
                    data JSONB NOT NULL,
                    features JSONB NOT NULL,
                    metadata JSONB,
                    timestamp TIMESTAMP NOT NULL,
                    confidence FLOAT NOT NULL,
                    frequency INTEGER NOT NULL,
                    tags TEXT[]
                );
                
                CREATE INDEX IF NOT EXISTS idx_patterns_type_time 
                ON patterns (pattern_type, timestamp);
                
                CREATE INDEX IF NOT EXISTS idx_patterns_category 
                ON patterns (category);
                
                CREATE INDEX IF NOT EXISTS idx_patterns_frequency 
                ON patterns (frequency DESC);
            """)

    async def _initialize_sqlite_tables(self):
        """Initialize SQLite tables."""
        await self.sqlite_conn.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT NOT NULL,
                category TEXT NOT NULL,
                data TEXT NOT NULL,
                features TEXT NOT NULL,
                metadata TEXT,
                timestamp TEXT NOT NULL,
                confidence REAL NOT NULL,
                frequency INTEGER NOT NULL,
                tags TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_patterns_type_time 
            ON patterns (pattern_type, timestamp);
            
            CREATE INDEX IF NOT EXISTS idx_patterns_category 
            ON patterns (category);
            
            CREATE INDEX IF NOT EXISTS idx_patterns_frequency 
            ON patterns (frequency DESC);
        """)
        await self.sqlite_conn.commit()

# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'postgres_url': 'postgresql://user:password@localhost/dbname',
        'sqlite_path': 'patterns.db',
        'window_size': 60,
        'min_pattern_length': 10,
        'similarity_threshold': 0.8,
        'clustering_eps': 0.3,
        'clustering_min_samples': 5,
        'pattern_retention_days': 30
    }
    
    # Initialize pattern library
    library = PatternLibrary(config)
    
    # Example pattern management
    async def main():
        # Example workload pattern
        pattern_data = {
            'values': [10, 15, 25, 40, 60, 45, 30, 20, 15, 10],
            'source': 'web_traffic',
            'region': 'us-west'
        }
        
        # Add pattern
        pattern = await library.add_pattern(
            PatternType.WORKLOAD,
            pattern_data,
            {'source_system': 'web_server'},
            {'high_traffic', 'spike'}
        )
        
        if pattern:
            print(f"Added pattern: {pattern.pattern_id}")
            print(f"Category: {pattern.category.value}")
            print(f"Features:", json.dumps(pattern.features, indent=2))
        
        # Get patterns
        patterns = await library.get_patterns(
            pattern_type=PatternType.WORKLOAD,
            category=PatternCategory.SPIKE,
            tags={'high_traffic'},
            min_confidence=0.7
        )
        
        print(f"\nFound {len(patterns)} matching patterns")
        
        # Analyze patterns
        analysis = await library.analyze_patterns(PatternType.WORKLOAD)
        print("\nPattern Analysis:")
        print(json.dumps(analysis, indent=2))
    
    # Run example
    asyncio.run(main())