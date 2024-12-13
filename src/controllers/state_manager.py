from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
import asyncio
import logging
import json
import aioredis
import aiokafka
from enum import Enum
import pickle
import zlib
import hashlib
from collections import defaultdict
import aiosqlite
import asyncpg

class StateType(Enum):
    """Types of managed states."""
    SYSTEM = "system"
    SCALING = "scaling"
    WORKLOAD = "workload"
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    CONFIGURATION = "configuration"
    INCIDENT = "incident"

@dataclass
class StateSnapshot:
    """Container for state snapshots."""
    snapshot_id: str
    state_type: StateType
    timestamp: datetime
    data: Dict
    metadata: Dict = field(default_factory=dict)
    checksum: str = ""
    version: int = 1
    parent_id: Optional[str] = None

    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate checksum of state data."""
        data_str = json.dumps(self.data, sort_keys=True)
        return hashlib.sha256(data_str.encode('utf-8')).hexdigest()

class StateManager:
    """
    Manages system state, including persistence, versioning,
    and synchronization across distributed components.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the state manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.node_id = config['node_id']
        self.cluster_nodes = set(config['cluster_nodes'])
        
        # State storage
        self.current_state: Dict[StateType, Dict] = defaultdict(dict)
        self.state_history: Dict[StateType, List[StateSnapshot]] = defaultdict(list)
        self.state_locks: Dict[StateType, asyncio.Lock] = {}
        
        # Communication channels
        self.redis_pool = None
        self.kafka_producer = None
        self.kafka_consumer = None
        self.postgres_pool = None
        self.sqlite_conn = None
        
        # Initialize locks for each state type
        for state_type in StateType:
            self.state_locks[state_type] = asyncio.Lock()
        
        # State versioning
        self.state_versions: Dict[StateType, int] = defaultdict(int)
        
        # Caching
        self.state_cache = {}
        self.cache_ttl = config.get('cache_ttl', 300)  # 5 minutes
        
        # Start background tasks
        asyncio.create_task(self._initialize_storage())
        asyncio.create_task(self._state_sync_loop())
        asyncio.create_task(self._snapshot_cleanup_loop())
        asyncio.create_task(self._cache_cleanup_loop())

    async def _initialize_storage(self):
        """Initialize storage connections."""
        try:
            # Initialize Redis
            self.redis_pool = await aioredis.create_redis_pool(
                self.config['redis_url'],
                minsize=5,
                maxsize=10
            )
            
            # Initialize Kafka
            self.kafka_producer = aiokafka.AIOKafkaProducer(
                bootstrap_servers=self.config['kafka_brokers']
            )
            await self.kafka_producer.start()
            
            self.kafka_consumer = aiokafka.AIOKafkaConsumer(
                self.config['state_topic'],
                bootstrap_servers=self.config['kafka_brokers'],
                group_id=f"state_group_{self.node_id}"
            )
            await self.kafka_consumer.start()
            
            # Initialize PostgreSQL
            if 'postgres_url' in self.config:
                self.postgres_pool = await asyncpg.create_pool(
                    self.config['postgres_url'],
                    min_size=5,
                    max_size=10
                )
                
                # Create tables if needed
                await self._initialize_postgres_tables()
            
            # Initialize SQLite
            if 'sqlite_path' in self.config:
                self.sqlite_conn = await aiosqlite.connect(
                    self.config['sqlite_path']
                )
                await self._initialize_sqlite_tables()
            
            # Load initial state
            await self._load_initial_state()
            
        except Exception as e:
            self.logger.error(f"Error initializing storage: {str(e)}")
            raise

    async def update_state(
        self,
        state_type: StateType,
        data: Dict,
        metadata: Optional[Dict] = None
    ) -> StateSnapshot:
        """
        Update state for a given type.
        
        Args:
            state_type: Type of state to update
            data: State data
            metadata: Optional metadata
            
        Returns:
            StateSnapshot object
        """
        try:
            async with self.state_locks[state_type]:
                # Create snapshot
                snapshot = await self._create_snapshot(
                    state_type,
                    data,
                    metadata
                )
                
                # Update current state
                self.current_state[state_type] = data.copy()
                
                # Update version
                self.state_versions[state_type] += 1
                
                # Store snapshot
                await self._store_snapshot(snapshot)
                
                # Update cache
                self.state_cache[state_type] = {
                    'data': data.copy(),
                    'timestamp': datetime.utcnow()
                }
                
                # Broadcast update
                await self._broadcast_state_update(snapshot)
                
                return snapshot
                
        except Exception as e:
            self.logger.error(f"Error updating state: {str(e)}")
            raise

    async def get_state(
        self,
        state_type: StateType,
        version: Optional[int] = None
    ) -> Dict:
        """
        Get state for a given type.
        
        Args:
            state_type: Type of state to get
            version: Optional specific version to retrieve
            
        Returns:
            State data
        """
        try:
            # Check cache first
            if version is None and state_type in self.state_cache:
                cache_entry = self.state_cache[state_type]
                if (datetime.utcnow() - cache_entry['timestamp']).total_seconds() < self.cache_ttl:
                    return cache_entry['data']
            
            async with self.state_locks[state_type]:
                if version is not None:
                    # Get specific version
                    snapshot = await self._get_snapshot_by_version(
                        state_type,
                        version
                    )
                    return snapshot.data if snapshot else {}
                
                # Get current state
                return self.current_state[state_type].copy()
                
        except Exception as e:
            self.logger.error(f"Error getting state: {str(e)}")
            return {}

    async def get_state_history(
        self,
        state_type: StateType,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[StateSnapshot]:
        """Get state history for a given type."""
        try:
            snapshots = []
            
            if self.postgres_pool:
                # Get from PostgreSQL
                query = """
                    SELECT * FROM state_snapshots
                    WHERE state_type = $1
                """
                params = [state_type.value]
                
                if start_time:
                    query += " AND timestamp >= $2"
                    params.append(start_time)
                
                if end_time:
                    query += " AND timestamp <= $3"
                    params.append(end_time)
                
                query += " ORDER BY timestamp DESC"
                
                if limit:
                    query += f" LIMIT {limit}"
                
                async with self.postgres_pool.acquire() as conn:
                    rows = await conn.fetch(query, *params)
                    
                    for row in rows:
                        snapshots.append(StateSnapshot(
                            snapshot_id=row['snapshot_id'],
                            state_type=StateType(row['state_type']),
                            timestamp=row['timestamp'],
                            data=json.loads(row['data']),
                            metadata=json.loads(row['metadata']),
                            checksum=row['checksum'],
                            version=row['version'],
                            parent_id=row['parent_id']
                        ))
            
            return snapshots
            
        except Exception as e:
            self.logger.error(f"Error getting state history: {str(e)}")
            return []

    async def _create_snapshot(
        self,
        state_type: StateType,
        data: Dict,
        metadata: Optional[Dict] = None
    ) -> StateSnapshot:
        """Create a new state snapshot."""
        snapshot_id = self._generate_snapshot_id(state_type, data)
        
        snapshot = StateSnapshot(
            snapshot_id=snapshot_id,
            state_type=state_type,
            timestamp=datetime.utcnow(),
            data=data.copy(),
            metadata=metadata or {},
            version=self.state_versions[state_type] + 1,
            parent_id=self._get_latest_snapshot_id(state_type)
        )
        
        return snapshot

    async def _store_snapshot(
        self,
        snapshot: StateSnapshot
    ):
        """Store state snapshot."""
        try:
            # Store in memory
            self.state_history[snapshot.state_type].append(snapshot)
            
            # Store in PostgreSQL if available
            if self.postgres_pool:
                async with self.postgres_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO state_snapshots (
                            snapshot_id, state_type, timestamp, data, metadata,
                            checksum, version, parent_id
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    snapshot.snapshot_id,
                    snapshot.state_type.value,
                    snapshot.timestamp,
                    json.dumps(snapshot.data),
                    json.dumps(snapshot.metadata),
                    snapshot.checksum,
                    snapshot.version,
                    snapshot.parent_id
                    )
            
            # Store in SQLite if available
            if self.sqlite_conn:
                await self.sqlite_conn.execute("""
                    INSERT INTO state_snapshots (
                        snapshot_id, state_type, timestamp, data, metadata,
                        checksum, version, parent_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.snapshot_id,
                    snapshot.state_type.value,
                    snapshot.timestamp.isoformat(),
                    json.dumps(snapshot.data),
                    json.dumps(snapshot.metadata),
                    snapshot.checksum,
                    snapshot.version,
                    snapshot.parent_id
                ))
                await self.sqlite_conn.commit()
            
            # Store in Redis for quick access
            compressed_data = zlib.compress(pickle.dumps(snapshot))
            await self.redis_pool.setex(
                f"state:{snapshot.state_type.value}:{snapshot.version}",
                self.config.get('redis_ttl', 3600),
                compressed_data
            )
            
        except Exception as e:
            self.logger.error(f"Error storing snapshot: {str(e)}")
            raise

    async def _broadcast_state_update(
        self,
        snapshot: StateSnapshot
    ):
        """Broadcast state update to other nodes."""
        try:
            message = {
                'type': 'state_update',
                'snapshot_id': snapshot.snapshot_id,
                'state_type': snapshot.state_type.value,
                'timestamp': snapshot.timestamp.isoformat(),
                'data': snapshot.data,
                'metadata': snapshot.metadata,
                'version': snapshot.version,
                'node_id': self.node_id
            }
            
            await self.kafka_producer.send_and_wait(
                self.config['state_topic'],
                json.dumps(message).encode('utf-8')
            )
            
        except Exception as e:
            self.logger.error(f"Error broadcasting state update: {str(e)}")

    async def _state_sync_loop(self):
        """Background task for state synchronization."""
        try:
            async for message in self.kafka_consumer:
                try:
                    data = json.loads(message.value.decode('utf-8'))
                    
                    if data['type'] == 'state_update' and data['node_id'] != self.node_id:
                        await self._handle_state_update(data)
                        
                except Exception as e:
                    self.logger.error(f"Error processing state update: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Error in state sync loop: {str(e)}")
            await asyncio.sleep(1)
            asyncio.create_task(self._state_sync_loop())

    async def _handle_state_update(
        self,
        data: Dict
    ):
        """Handle incoming state update."""
        try:
            state_type = StateType(data['state_type'])
            
            # Check version
            if data['version'] <= self.state_versions[state_type]:
                return
            
            # Create snapshot
            snapshot = StateSnapshot(
                snapshot_id=data['snapshot_id'],
                state_type=state_type,
                timestamp=datetime.fromisoformat(data['timestamp']),
                data=data['data'],
                metadata=data['metadata'],
                version=data['version']
            )
            
            # Update state
            async with self.state_locks[state_type]:
                self.current_state[state_type] = data['data']
                self.state_versions[state_type] = data['version']
                
                # Store snapshot
                await self._store_snapshot(snapshot)
                
                # Update cache
                self.state_cache[state_type] = {
                    'data': data['data'],
                    'timestamp': datetime.utcnow()
                }
                
        except Exception as e:
            self.logger.error(f"Error handling state update: {str(e)}")

    def _generate_snapshot_id(
        self,
        state_type: StateType,
        data: Dict
    ) -> str:
        """Generate unique snapshot ID."""
        data_str = f"{state_type.value}:{json.dumps(data, sort_keys=True)}:{datetime.utcnow().isoformat()}"
        return hashlib.sha256(data_str.encode('utf-8')).hexdigest()

    def _get_latest_snapshot_id(
        self,
        state_type: StateType
    ) -> Optional[str]:
        """Get ID of latest snapshot for a state type."""
        if self.state_history[state_type]:
            return self.state_history[state_type][-1].snapshot_id
        return None

    async def _initialize_postgres_tables(self):
        """Initialize PostgreSQL tables."""
        async with self.postgres_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS state_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    state_type TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    data JSONB NOT NULL,
                    metadata JSONB,
                    checksum TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    parent_id TEXT,
                    FOREIGN KEY (parent_id) REFERENCES state_snapshots (snapshot_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_state_snapshots_type_time 
                ON state_snapshots (state_type, timestamp);
            """)

    async def _initialize_sqlite_tables(self):
        """Initialize SQLite tables."""
        await self.sqlite_conn.execute("""
            CREATE TABLE IF NOT EXISTS state_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                state_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                data TEXT NOT NULL,
                metadata TEXT,
                checksum TEXT NOT NULL,
                version INTEGER NOT NULL,
                parent_id TEXT,
                FOREIGN KEY (parent_id) REFERENCES state_snapshots (snapshot_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_state_snapshots_type_time 
            ON state_snapshots (state_type, timestamp);
        """)
        await self.sqlite_conn.commit()

    async def _load_initial_state(self):
        """Load initial state from storage."""
        try:
            # Try to load from PostgreSQL first
            if self.postgres_pool:
                async with self.postgres_pool.acquire() as conn:
                    for state_type in StateType:
                        row = await conn.fetchrow("""
                            SELECT * FROM state_snapshots
                            WHERE state_type = $1
                            ORDER BY version DESC
                            LIMIT 1
                        """, state_type.value)
                        
                        if row:
                            self.current_state[state_type] = json.loads(row['data'])
                            self.state_versions[state_type] = row['version']
            
            # Fall back to SQLite if needed
            elif self.sqlite_conn:
                for state_type in StateType:
                    async with self.sqlite_conn.execute("""
                        SELECT * FROM state_snapshots
                        WHERE state_type = ?
                        ORDER BY version DESC
                        LIMIT 1
                    """, (state_type.value,)) as cursor:
                        row = await cursor.fetchone()
                        
                        if row:
                            self.current_state[state_type] = json.loads(row[3])  # data column
                            self.state_versions[state_type] = row[6]  # version column
            
        except Exception as e:
            self.logger.error(f"Error loading initial state: {str(e)}")

    async def _snapshot_cleanup_loop(self):
        """Background task for cleaning up old snapshots."""
        while True:
            try:
                # Get retention period
                retention_days = self.config.get('snapshot_retention_days', 30)
                cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
                
                # Clean up PostgreSQL
                if self.postgres_pool:
                    async with self.postgres_pool.acquire() as conn:
                        await conn.execute("""
                            DELETE FROM state_snapshots
                            WHERE timestamp < $1
                        """, cutoff_date)
                
                # Clean up SQLite
                if self.sqlite_conn:
                    await self.sqlite_conn.execute("""
                        DELETE FROM state_snapshots
                        WHERE timestamp < ?
                    """, (cutoff_date.isoformat(),))
                    await self.sqlite_conn.commit()
                
                # Clean up memory
                for state_type in StateType:
                    self.state_history[state_type] = [
                        snapshot for snapshot in self.state_history[state_type]
                        if snapshot.timestamp > cutoff_date
                    ]
                
                # Sleep before next cleanup
                await asyncio.sleep(
                    self.config.get('cleanup_interval', 86400)  # 24 hours
                )
                
            except Exception as e:
                self.logger.error(f"Error in snapshot cleanup: {str(e)}")
                await asyncio.sleep(3600)

    async def _cache_cleanup_loop(self):
        """Background task for cleaning up expired cache entries."""
        while True:
            try:
                current_time = datetime.utcnow()
                
                # Clean up expired cache entries
                expired_keys = [
                    state_type for state_type, entry in self.state_cache.items()
                    if (current_time - entry['timestamp']).total_seconds() > self.cache_ttl
                ]
                
                for key in expired_keys:
                    del self.state_cache[key]
                
                # Sleep before next cleanup
                await asyncio.sleep(
                    self.config.get('cache_cleanup_interval', 60)
                )
                
            except Exception as e:
                self.logger.error(f"Error in cache cleanup: {str(e)}")
                await asyncio.sleep(60)

# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'node_id': 'node1',
        'cluster_nodes': ['node1', 'node2', 'node3'],
        'redis_url': 'redis://localhost',
        'kafka_brokers': ['localhost:9092'],
        'state_topic': 'system_state',
        'postgres_url': 'postgresql://user:password@localhost/dbname',
        'sqlite_path': 'state.db',
        'cache_ttl': 300,
        'snapshot_retention_days': 30
    }
    
    # Initialize state manager
    state_manager = StateManager(config)
    
    # Example state management
    async def main():
        # Update system state
        system_state = {
            'status': 'healthy',
            'load': 0.7,
            'memory_usage': 0.6,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        snapshot = await state_manager.update_state(
            StateType.SYSTEM,
            system_state,
            {'source': 'monitoring_system'}
        )
        
        print(f"Created snapshot: {snapshot.snapshot_id}")
        
        # Get current state
        current_state = await state_manager.get_state(StateType.SYSTEM)
        print("Current state:", json.dumps(current_state, indent=2))
        
        # Get state history
        history = await state_manager.get_state_history(
            StateType.SYSTEM,
            start_time=datetime.utcnow() - timedelta(hours=1),
            limit=10
        )
        
        print(f"Found {len(history)} historical snapshots")
        
        for snapshot in history:
            print(f"Snapshot {snapshot.snapshot_id} at {snapshot.timestamp}")
    
    # Run example
    asyncio.run(main())