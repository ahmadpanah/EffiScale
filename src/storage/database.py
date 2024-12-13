from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import asyncio
import logging
import json
from enum import Enum
from dataclasses import dataclass
import aiopg
import aiomysql
import aiosqlite
import motor.motor_asyncio
from pymongo import ASCENDING, DESCENDING
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text
import sqlalchemy as sa
from contextlib import asynccontextmanager

class DatabaseType(Enum):
    """Supported database types."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"
    REDIS = "redis"

class QueryType(Enum):
    """Query types."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    AGGREGATE = "aggregate"

@dataclass
class DatabaseConfig:
    """Database configuration."""
    db_type: DatabaseType
    host: Optional[str] = None
    port: Optional[int] = None
    database: str = "main"
    username: Optional[str] = None
    password: Optional[str] = None
    pool_size: int = 5
    max_overflow: int = 10
    pool_recycle: int = 3600
    ssl: bool = False
    timeout: int = 30

class DatabaseManager:
    """
    Manages database connections and operations.
    Supports multiple database backends and connection pooling.
    """
    
    def __init__(self, config: Dict[str, DatabaseConfig]):
        """
        Initialize database manager.
        
        Args:
            config: Dictionary of database configurations
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize connections
        self.connections: Dict[str, Any] = {}
        self.engines: Dict[str, Any] = {}
        self.pools: Dict[str, Any] = {}
        
        # Initialize session factories
        self.session_factories: Dict[str, sessionmaker] = {}
        
        # Query tracking
        self.query_stats: Dict[str, Dict] = {}
        
        # Start background tasks
        asyncio.create_task(self._initialize_connections())
        asyncio.create_task(self._monitor_connections())

    async def execute_query(
        self,
        db_name: str,
        query: str,
        params: Optional[Union[Dict, List]] = None,
        query_type: QueryType = QueryType.SELECT,
        timeout: Optional[int] = None
    ) -> Union[List[Dict], Dict, None]:
        """
        Execute database query.
        
        Args:
            db_name: Database name
            query: Query string
            params: Query parameters
            query_type: Type of query
            timeout: Query timeout
            
        Returns:
            Query results
        """
        try:
            db_config = self.config[db_name]
            start_time = datetime.utcnow()
            
            if db_config.db_type in {DatabaseType.POSTGRESQL, DatabaseType.MYSQL}:
                return await self._execute_sql_query(
                    db_name,
                    query,
                    params,
                    query_type,
                    timeout
                )
            elif db_config.db_type == DatabaseType.MONGODB:
                return await self._execute_mongo_query(
                    db_name,
                    query,
                    params,
                    query_type
                )
            elif db_config.db_type == DatabaseType.REDIS:
                return await self._execute_redis_command(
                    db_name,
                    query,
                    params
                )
            else:
                raise ValueError(f"Unsupported database type: {db_config.db_type}")
            
        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            raise

    async def execute_transaction(
        self,
        db_name: str,
        queries: List[Tuple[str, Optional[Dict]]],
        timeout: Optional[int] = None
    ) -> bool:
        """Execute multiple queries in transaction."""
        try:
            db_config = self.config[db_name]
            
            if db_config.db_type in {DatabaseType.POSTGRESQL, DatabaseType.MYSQL}:
                async with self._get_sql_transaction(db_name) as transaction:
                    for query, params in queries:
                        await transaction.execute(text(query), params or {})
                    return True
                    
            elif db_config.db_type == DatabaseType.MONGODB:
                async with await self.connections[db_name].start_session() as session:
                    async with session.start_transaction():
                        for query, params in queries:
                            await self._execute_mongo_query(
                                db_name,
                                query,
                                params,
                                QueryType.INSERT
                            )
                    return True
                    
            else:
                raise ValueError(f"Transactions not supported for {db_config.db_type}")
            
        except Exception as e:
            self.logger.error(f"Error executing transaction: {str(e)}")
            return False

    async def execute_batch(
        self,
        db_name: str,
        query: str,
        params_list: List[Dict],
        batch_size: int = 1000
    ) -> bool:
        """Execute batch operation."""
        try:
            db_config = self.config[db_name]
            
            # Process in batches
            for i in range(0, len(params_list), batch_size):
                batch = params_list[i:i + batch_size]
                
                if db_config.db_type in {DatabaseType.POSTGRESQL, DatabaseType.MYSQL}:
                    async with self._get_sql_transaction(db_name) as transaction:
                        await transaction.execute(
                            text(query),
                            batch
                        )
                        
                elif db_config.db_type == DatabaseType.MONGODB:
                    if query.startswith("insert"):
                        await self.connections[db_name].insert_many(batch)
                    else:
                        for params in batch:
                            await self._execute_mongo_query(
                                db_name,
                                query,
                                params,
                                QueryType.INSERT
                            )
                            
                else:
                    raise ValueError(f"Batch operations not supported for {db_config.db_type}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing batch: {str(e)}")
            return False

    @asynccontextmanager
    async def get_connection(
        self,
        db_name: str
    ):
        """Get database connection."""
        try:
            db_config = self.config[db_name]
            
            if db_config.db_type in {DatabaseType.POSTGRESQL, DatabaseType.MYSQL}:
                async with self.pools[db_name].acquire() as connection:
                    yield connection
                    
            elif db_config.db_type == DatabaseType.MONGODB:
                yield self.connections[db_name]
                
            elif db_config.db_type == DatabaseType.REDIS:
                yield self.connections[db_name]
                
            else:
                raise ValueError(f"Unsupported database type: {db_config.db_type}")
            
        except Exception as e:
            self.logger.error(f"Error getting connection: {str(e)}")
            raise

    async def _initialize_connections(self):
        """Initialize database connections."""
        try:
            for db_name, db_config in self.config.items():
                if db_config.db_type == DatabaseType.POSTGRESQL:
                    # PostgreSQL connection
                    self.pools[db_name] = await aiopg.create_pool(
                        database=db_config.database,
                        user=db_config.username,
                        password=db_config.password,
                        host=db_config.host,
                        port=db_config.port,
                        minsize=1,
                        maxsize=db_config.pool_size,
                        timeout=db_config.timeout
                    )
                    
                    # SQLAlchemy engine
                    self.engines[db_name] = create_async_engine(
                        f"postgresql+asyncpg://{db_config.username}:{db_config.password}@"
                        f"{db_config.host}:{db_config.port}/{db_config.database}",
                        pool_size=db_config.pool_size,
                        max_overflow=db_config.max_overflow,
                        pool_recycle=db_config.pool_recycle
                    )
                    
                elif db_config.db_type == DatabaseType.MYSQL:
                    # MySQL connection
                    self.pools[db_name] = await aiomysql.create_pool(
                        db=db_config.database,
                        user=db_config.username,
                        password=db_config.password,
                        host=db_config.host,
                        port=db_config.port,
                        minsize=1,
                        maxsize=db_config.pool_size,
                        autocommit=True
                    )
                    
                elif db_config.db_type == DatabaseType.MONGODB:
                    # MongoDB connection
                    client = motor.motor_asyncio.AsyncIOMotorClient(
                        f"mongodb://{db_config.username}:{db_config.password}@"
                        f"{db_config.host}:{db_config.port}"
                    )
                    self.connections[db_name] = client[db_config.database]
                    
                elif db_config.db_type == DatabaseType.REDIS:
                    # Redis connection
                    self.connections[db_name] = await aioredis.from_url(
                        f"redis://{db_config.host}:{db_config.port}",
                        db=0,
                        password=db_config.password,
                        max_connections=db_config.pool_size
                    )
                    
                # Initialize session factory
                if db_config.db_type in {DatabaseType.POSTGRESQL, DatabaseType.MYSQL}:
                    self.session_factories[db_name] = sessionmaker(
                        self.engines[db_name],
                        class_=AsyncSession,
                        expire_on_commit=False
                    )
            
        except Exception as e:
            self.logger.error(f"Error initializing connections: {str(e)}")
            raise

    async def _monitor_connections(self):
        """Monitor database connections."""
        while True:
            try:
                for db_name, db_config in self.config.items():
                    # Check connection status
                    if db_config.db_type in {DatabaseType.POSTGRESQL, DatabaseType.MYSQL}:
                        pool = self.pools[db_name]
                        self.logger.info(
                            f"Pool {db_name}: "
                            f"size={pool.size}, "
                            f"freesize={pool.freesize}"
                        )
                        
                    elif db_config.db_type == DatabaseType.MONGODB:
                        # Ping MongoDB
                        await self.connections[db_name].command('ping')
                        
                    elif db_config.db_type == DatabaseType.REDIS:
                        # Ping Redis
                        await self.connections[db_name].ping()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error monitoring connections: {str(e)}")
                await asyncio.sleep(60)

    async def _execute_sql_query(
        self,
        db_name: str,
        query: str,
        params: Optional[Dict] = None,
        query_type: QueryType = QueryType.SELECT,
        timeout: Optional[int] = None
    ) -> Union[List[Dict], Dict, None]:
        """Execute SQL query."""
        try:
            async with self._get_sql_transaction(db_name) as transaction:
                result = await transaction.execute(
                    text(query),
                    params or {}
                )
                
                if query_type == QueryType.SELECT:
                    rows = await result.fetchall()
                    return [dict(row) for row in rows]
                    
                elif query_type == QueryType.INSERT:
                    return {"lastrowid": result.lastrowid}
                    
                elif query_type == QueryType.UPDATE:
                    return {"rowcount": result.rowcount}
                    
                elif query_type == QueryType.DELETE:
                    return {"rowcount": result.rowcount}
                    
                return None
                
        except Exception as e:
            self.logger.error(f"Error executing SQL query: {str(e)}")
            raise

    async def _execute_mongo_query(
        self,
        db_name: str,
        query: str,
        params: Optional[Dict] = None,
        query_type: QueryType = QueryType.SELECT
    ) -> Union[List[Dict], Dict, None]:
        """Execute MongoDB query."""
        try:
            collection = self.connections[db_name]
            params = params or {}
            
            if query_type == QueryType.SELECT:
                cursor = collection.find(params)
                return await cursor.to_list(None)
                
            elif query_type == QueryType.INSERT:
                result = await collection.insert_one(params)
                return {"inserted_id": str(result.inserted_id)}
                
            elif query_type == QueryType.UPDATE:
                result = await collection.update_many(
                    params.get("filter", {}),
                    params.get("update", {})
                )
                return {"modified_count": result.modified_count}
                
            elif query_type == QueryType.DELETE:
                result = await collection.delete_many(params)
                return {"deleted_count": result.deleted_count}
                
            elif query_type == QueryType.AGGREGATE:
                pipeline = params.get("pipeline", [])
                cursor = collection.aggregate(pipeline)
                return await cursor.to_list(None)
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error executing MongoDB query: {str(e)}")
            raise

    async def _execute_redis_command(
        self,
        db_name: str,
        command: str,
        params: Optional[List] = None
    ) -> Any:
        """Execute Redis command."""
        try:
            redis = self.connections[db_name]
            params = params or []
            
            # Execute command
            result = await redis.execute_command(command, *params)
            
            # Decode bytes to string if needed
            if isinstance(result, bytes):
                return result.decode('utf-8')
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing Redis command: {str(e)}")
            raise

    @asynccontextmanager
    async def _get_sql_transaction(self, db_name: str):
        """Get SQL transaction."""
        async with self.session_factories[db_name]() as session:
            async with session.begin():
                yield session

# Example usage
if __name__ == "__main__":
    # Database configurations
    configs = {
        "postgres_main": DatabaseConfig(
            db_type=DatabaseType.POSTGRESQL,
            host="localhost",
            port=5432,
            database="myapp",
            username="user",
            password="pass"
        ),
        "mongo_logs": DatabaseConfig(
            db_type=DatabaseType.MONGODB,
            host="localhost",
            port=27017,
            database="logs",
            username="user",
            password="pass"
        ),
        "redis_cache": DatabaseConfig(
            db_type=DatabaseType.REDIS,
            host="localhost",
            port=6379,
            password="pass"
        )
    }
    
    # Initialize manager
    db_manager = DatabaseManager(configs)
    
    # Example operations
    async def main():
        # SQL query
        users = await db_manager.execute_query(
            "postgres_main",
            "SELECT * FROM users WHERE age > :age",
            {"age": 18}
        )
        print("Users:", users)
        
        # MongoDB query
        logs = await db_manager.execute_query(
            "mongo_logs",
            "find",
            {"level": "ERROR"},
            QueryType.SELECT
        )
        print("Error logs:", logs)
        
        # Redis command
        await db_manager.execute_query(
            "redis_cache",
            "SET",
            ["key", "value"]
        )
        
        value = await db_manager.execute_query(
            "redis_cache",
            "GET",
            ["key"]
        )
        print("Redis value:", value)
        
        # Transaction example
        success = await db_manager.execute_transaction(
            "postgres_main",
            [
                ("INSERT INTO users (name, age) VALUES (:name, :age)", 
                 {"name": "John", "age": 30}),
                ("UPDATE users SET status = :status WHERE name = :name",
                 {"status": "active", "name": "John"})
            ]
        )
        print("Transaction success:", success)
        
        # Batch operation
        users_data = [
            {"name": f"User{i}", "age": 20 + i}
            for i in range(100)
        ]
        
        success = await db_manager.execute_batch(
            "postgres_main",
            "INSERT INTO users (name, age) VALUES (:name, :age)",
            users_data
        )
        print("Batch insert success:", success)
    
    # Run example
    asyncio.run(main())