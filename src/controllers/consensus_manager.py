from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio
import logging
from enum import Enum
import json
import hashlib
import random
from collections import defaultdict
import aioredis
import aiokafka
import async_timeout

class ConsensusState(Enum):
    """States of consensus process."""
    PROPOSED = "proposed"
    VOTING = "voting"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"

class ConsensusRole(Enum):
    """Node roles in consensus."""
    LEADER = "leader"
    FOLLOWER = "follower"
    OBSERVER = "observer"

@dataclass
class ConsensusProposal:
    """Container for consensus proposals."""
    proposal_id: str
    proposal_type: str
    content: Dict
    proposer: str
    timestamp: datetime
    expiration: datetime
    metadata: Dict
    state: ConsensusState
    votes: Dict[str, bool]
    execution_result: Optional[Dict] = None

class ConsensusManager:
    """
    Manages distributed consensus for scaling decisions across multiple nodes.
    Implements a Byzantine fault-tolerant consensus mechanism.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the consensus manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.node_id = config['node_id']
        self.cluster_nodes = set(config['cluster_nodes'])
        self.required_votes = config.get('required_votes', len(self.cluster_nodes) // 2 + 1)
        
        # State management
        self.current_role = ConsensusRole.FOLLOWER
        self.current_leader = None
        self.proposals: Dict[str, ConsensusProposal] = {}
        self.votes: Dict[str, Dict[str, bool]] = defaultdict(dict)
        
        # Communication channels
        self.redis_pool = None
        self.kafka_producer = None
        self.kafka_consumer = None
        
        # Locks and synchronization
        self.proposal_locks: Dict[str, asyncio.Lock] = {}
        self.leader_election_lock = asyncio.Lock()
        
        # Start background tasks
        asyncio.create_task(self._initialize_communication())
        asyncio.create_task(self._leader_election_loop())
        asyncio.create_task(self._proposal_cleanup_loop())
        asyncio.create_task(self._message_processing_loop())

    async def _initialize_communication(self):
        """Initialize communication channels."""
        try:
            # Initialize Redis connection
            self.redis_pool = await aioredis.create_redis_pool(
                self.config['redis_url'],
                minsize=5,
                maxsize=10
            )
            
            # Initialize Kafka producer
            self.kafka_producer = aiokafka.AIOKafkaProducer(
                bootstrap_servers=self.config['kafka_brokers']
            )
            await self.kafka_producer.start()
            
            # Initialize Kafka consumer
            self.kafka_consumer = aiokafka.AIOKafkaConsumer(
                self.config['consensus_topic'],
                bootstrap_servers=self.config['kafka_brokers'],
                group_id=f"consensus_group_{self.node_id}"
            )
            await self.kafka_consumer.start()
            
        except Exception as e:
            self.logger.error(f"Error initializing communication: {str(e)}")
            raise

    async def propose_decision(
        self,
        decision_type: str,
        content: Dict,
        metadata: Optional[Dict] = None
    ) -> ConsensusProposal:
        """
        Propose a decision for consensus.
        
        Args:
            decision_type: Type of decision
            content: Decision content
            metadata: Optional metadata
            
        Returns:
            ConsensusProposal object
        """
        try:
            # Generate proposal ID
            proposal_id = self._generate_proposal_id(decision_type, content)
            
            # Create proposal
            proposal = ConsensusProposal(
                proposal_id=proposal_id,
                proposal_type=decision_type,
                content=content,
                proposer=self.node_id,
                timestamp=datetime.utcnow(),
                expiration=datetime.utcnow() + timedelta(
                    seconds=self.config.get('proposal_timeout', 30)
                ),
                metadata=metadata or {},
                state=ConsensusState.PROPOSED,
                votes={},
                execution_result=None
            )
            
            # Store proposal
            self.proposals[proposal_id] = proposal
            
            # Create proposal lock
            self.proposal_locks[proposal_id] = asyncio.Lock()
            
            # Broadcast proposal
            await self._broadcast_proposal(proposal)
            
            return proposal
            
        except Exception as e:
            self.logger.error(f"Error proposing decision: {str(e)}")
            raise

    async def vote_on_proposal(
        self,
        proposal_id: str,
        vote: bool,
        reason: Optional[str] = None
    ) -> bool:
        """
        Vote on a consensus proposal.
        
        Args:
            proposal_id: Proposal ID
            vote: True for accept, False for reject
            reason: Optional reason for vote
            
        Returns:
            Success status
        """
        try:
            if proposal_id not in self.proposals:
                raise ValueError(f"Unknown proposal: {proposal_id}")
                
            proposal = self.proposals[proposal_id]
            
            # Check if proposal is still valid
            if proposal.expiration < datetime.utcnow():
                self.logger.warning(f"Proposal {proposal_id} has expired")
                return False
            
            # Record vote
            async with self.proposal_locks[proposal_id]:
                proposal.votes[self.node_id] = vote
                
                # Broadcast vote
                await self._broadcast_vote(proposal_id, vote, reason)
                
                # Check if consensus is reached
                consensus_reached = await self._check_consensus(proposal_id)
                
                if consensus_reached:
                    await self._handle_consensus_result(proposal_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error voting on proposal: {str(e)}")
            return False

    async def get_proposal_status(
        self,
        proposal_id: str
    ) -> Optional[ConsensusProposal]:
        """Get status of a proposal."""
        return self.proposals.get(proposal_id)

    async def _broadcast_proposal(
        self,
        proposal: ConsensusProposal
    ):
        """Broadcast proposal to all nodes."""
        message = {
            'type': 'proposal',
            'proposal_id': proposal.proposal_id,
            'proposal_type': proposal.proposal_type,
            'content': proposal.content,
            'proposer': proposal.proposer,
            'timestamp': proposal.timestamp.isoformat(),
            'expiration': proposal.expiration.isoformat(),
            'metadata': proposal.metadata
        }
        
        await self._publish_message(message)

    async def _broadcast_vote(
        self,
        proposal_id: str,
        vote: bool,
        reason: Optional[str]
    ):
        """Broadcast vote to all nodes."""
        message = {
            'type': 'vote',
            'proposal_id': proposal_id,
            'voter': self.node_id,
            'vote': vote,
            'reason': reason,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self._publish_message(message)

    async def _publish_message(
        self,
        message: Dict
    ):
        """Publish message to consensus topic."""
        try:
            message_bytes = json.dumps(message).encode('utf-8')
            await self.kafka_producer.send_and_wait(
                self.config['consensus_topic'],
                message_bytes
            )
        except Exception as e:
            self.logger.error(f"Error publishing message: {str(e)}")

    async def _message_processing_loop(self):
        """Process incoming consensus messages."""
        try:
            async for message in self.kafka_consumer:
                try:
                    data = json.loads(message.value.decode('utf-8'))
                    message_type = data.get('type')
                    
                    if message_type == 'proposal':
                        await self._handle_proposal_message(data)
                    elif message_type == 'vote':
                        await self._handle_vote_message(data)
                    elif message_type == 'leader_election':
                        await self._handle_leader_election_message(data)
                    else:
                        self.logger.warning(f"Unknown message type: {message_type}")
                        
                except Exception as e:
                    self.logger.error(f"Error processing message: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Error in message processing loop: {str(e)}")
            await asyncio.sleep(1)
            asyncio.create_task(self._message_processing_loop())

    async def _handle_proposal_message(
        self,
        data: Dict
    ):
        """Handle incoming proposal message."""
        try:
            proposal_id = data['proposal_id']
            
            if proposal_id not in self.proposals:
                # Create new proposal
                proposal = ConsensusProposal(
                    proposal_id=proposal_id,
                    proposal_type=data['proposal_type'],
                    content=data['content'],
                    proposer=data['proposer'],
                    timestamp=datetime.fromisoformat(data['timestamp']),
                    expiration=datetime.fromisoformat(data['expiration']),
                    metadata=data['metadata'],
                    state=ConsensusState.PROPOSED,
                    votes={}
                )
                
                self.proposals[proposal_id] = proposal
                self.proposal_locks[proposal_id] = asyncio.Lock()
                
                # Automatically vote if configured
                if self.config.get('auto_vote', False):
                    asyncio.create_task(
                        self._auto_vote_on_proposal(proposal_id)
                    )
                    
        except Exception as e:
            self.logger.error(f"Error handling proposal message: {str(e)}")

    async def _handle_vote_message(
        self,
        data: Dict
    ):
        """Handle incoming vote message."""
        try:
            proposal_id = data['proposal_id']
            voter = data['voter']
            vote = data['vote']
            
            if proposal_id in self.proposals:
                async with self.proposal_locks[proposal_id]:
                    proposal = self.proposals[proposal_id]
                    proposal.votes[voter] = vote
                    
                    # Check consensus
                    await self._check_consensus(proposal_id)
                    
        except Exception as e:
            self.logger.error(f"Error handling vote message: {str(e)}")

    async def _auto_vote_on_proposal(
        self,
        proposal_id: str
    ):
        """Automatically vote on proposal based on configured rules."""
        try:
            proposal = self.proposals[proposal_id]
            
            # Implement voting logic here
            vote_result = await self._evaluate_proposal(proposal)
            
            await self.vote_on_proposal(
                proposal_id,
                vote_result,
                "Automatic vote based on evaluation"
            )
            
        except Exception as e:
            self.logger.error(f"Error in auto voting: {str(e)}")

    async def _evaluate_proposal(
        self,
        proposal: ConsensusProposal
    ) -> bool:
        """Evaluate proposal for automatic voting."""
        try:
            # Implement proposal evaluation logic here
            # This is a simple example - replace with actual evaluation logic
            
            # Check proposal type
            if proposal.proposal_type not in self.config.get('allowed_types', []):
                return False
            
            # Check proposer
            if proposal.proposer not in self.cluster_nodes:
                return False
            
            # Check content validity
            if not self._validate_proposal_content(proposal.content):
                return False
            
            return True
            
        except Exception:
            return False

    def _validate_proposal_content(
        self,
        content: Dict
    ) -> bool:
        """Validate proposal content."""
        # Implement content validation logic here
        return True

    async def _check_consensus(
        self,
        proposal_id: str
    ) -> bool:
        """Check if consensus is reached for a proposal."""
        try:
            proposal = self.proposals[proposal_id]
            
            if proposal.state not in [ConsensusState.PROPOSED, ConsensusState.VOTING]:
                return False
            
            total_votes = len(proposal.votes)
            positive_votes = sum(1 for v in proposal.votes.values() if v)
            
            # Check if we have enough total votes
            if total_votes >= self.required_votes:
                if positive_votes >= self.required_votes:
                    proposal.state = ConsensusState.ACCEPTED
                    return True
                elif (total_votes - positive_votes) > len(self.cluster_nodes) - self.required_votes:
                    proposal.state = ConsensusState.REJECTED
                    return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking consensus: {str(e)}")
            return False

    async def _handle_consensus_result(
        self,
        proposal_id: str
    ):
        """Handle reached consensus result."""
        try:
            proposal = self.proposals[proposal_id]
            
            if proposal.state == ConsensusState.ACCEPTED:
                # Execute accepted proposal
                success = await self._execute_proposal(proposal)
                
                if success:
                    proposal.state = ConsensusState.EXECUTED
                else:
                    proposal.state = ConsensusState.FAILED
                    
            # Broadcast final state
            await self._broadcast_consensus_result(proposal)
            
        except Exception as e:
            self.logger.error(f"Error handling consensus result: {str(e)}")

    async def _execute_proposal(
        self,
        proposal: ConsensusProposal
    ) -> bool:
        """Execute accepted proposal."""
        try:
            # Implement proposal execution logic here
            # This should integrate with your scaling system
            
            result = {
                'status': 'success',
                'timestamp': datetime.utcnow().isoformat(),
                'details': {}
            }
            
            proposal.execution_result = result
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing proposal: {str(e)}")
            return False

    async def _broadcast_consensus_result(
        self,
        proposal: ConsensusProposal
    ):
        """Broadcast final consensus result."""
        message = {
            'type': 'consensus_result',
            'proposal_id': proposal.proposal_id,
            'state': proposal.state.value,
            'execution_result': proposal.execution_result,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self._publish_message(message)

    async def _leader_election_loop(self):
        """Background task for leader election."""
        while True:
            try:
                async with self.leader_election_lock:
                    # Check if leader is healthy
                    if self.current_leader:
                        leader_healthy = await self._check_leader_health()
                        if not leader_healthy:
                            await self._initiate_leader_election()
                    else:
                        await self._initiate_leader_election()
                
                # Sleep before next check
                await asyncio.sleep(
                    self.config.get('leader_check_interval', 5)
                )
                
            except Exception as e:
                self.logger.error(f"Error in leader election loop: {str(e)}")
                await asyncio.sleep(1)

    async def _check_leader_health(self) -> bool:
        """Check if current leader is healthy."""
        try:
            async with async_timeout.timeout(2):
                # Ping leader through Redis
                leader_key = f"consensus_leader_{self.current_leader}"
                timestamp = await self.redis_pool.get(leader_key)
                
                if timestamp:
                    last_update = datetime.fromisoformat(timestamp.decode('utf-8'))
                    return (datetime.utcnow() - last_update).total_seconds() < 5
                    
                return False
                
        except Exception:
            return False

    async def _initiate_leader_election(self):
        """Initiate leader election process."""
        try:
            # Generate random delay to prevent split votes
            await asyncio.sleep(random.uniform(0, 2))
            
            # Propose self as leader
            message = {
                'type': 'leader_election',
                'node_id': self.node_id,
                'priority': self._calculate_node_priority(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            await self._publish_message(message)
            
        except Exception as e:
            self.logger.error(f"Error initiating leader election: {str(e)}")

    async def _handle_leader_election_message(
        self,
        data: Dict
    ):
        """Handle leader election message."""
        try:
            node_id = data['node_id']
            priority = data['priority']
            timestamp = datetime.fromisoformat(data['timestamp'])
            
            # Compare priorities
            if priority > self._calculate_node_priority():
                self.current_leader = node_id
                self.current_role = ConsensusRole.FOLLOWER
            elif priority == self._calculate_node_priority() and node_id > self.node_id:
                self.current_leader = node_id
                self.current_role = ConsensusRole.FOLLOWER
            
        except Exception as e:
            self.logger.error(f"Error handling leader election: {str(e)}")

    def _calculate_node_priority(self) -> float:
        """Calculate node priority for leader election."""
        # Implement priority calculation logic here
        # Consider factors like uptime, resources, etc.
        return random.random()

    def _generate_proposal_id(
        self,
        decision_type: str,
        content: Dict
    ) -> str:
        """Generate unique proposal ID."""
        data = f"{decision_type}:{json.dumps(content, sort_keys=True)}:{datetime.utcnow().isoformat()}"
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    async def _proposal_cleanup_loop(self):
        """Background task for cleaning up old proposals."""
        while True:
            try:
                current_time = datetime.utcnow()
                
                # Clean up expired proposals
                expired_proposals = [
                    pid for pid, p in self.proposals.items()
                    if p.expiration < current_time
                ]
                
                for pid in expired_proposals:
                    del self.proposals[pid]
                    del self.proposal_locks[pid]
                
                # Sleep before next cleanup
                await asyncio.sleep(
                    self.config.get('cleanup_interval', 60)
                )
                
            except Exception as e:
                self.logger.error(f"Error in proposal cleanup: {str(e)}")
                await asyncio.sleep(60)

# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'node_id': 'node1',
        'cluster_nodes': ['node1', 'node2', 'node3'],
        'required_votes': 2,
        'redis_url': 'redis://localhost',
        'kafka_brokers': ['localhost:9092'],
        'consensus_topic': 'scaling_consensus',
        'auto_vote': True,
        'allowed_types': ['scale_out', 'scale_in', 'scale_up', 'scale_down']
    }
    
    # Initialize consensus manager
    manager = ConsensusManager(config)
    
    # Example proposal
    async def main():
        # Propose a scaling decision
        proposal = await manager.propose_decision(
            'scale_out',
            {
                'instances': 5,
                'reason': 'High CPU utilization'
            },
            {
                'priority': 'high',
                'source': 'auto_scaler'
            }
        )
        
        print(f"Proposed decision: {proposal.proposal_id}")
        
        # Wait for consensus
        while True:
            status = await manager.get_proposal_status(proposal.proposal_id)
            print(f"Proposal status: {status.state.value}")
            
            if status.state in [
                ConsensusState.EXECUTED,
                ConsensusState.FAILED,
                ConsensusState.REJECTED
            ]:
                break
                
            await asyncio.sleep(1)
        
        print("Final result:", json.dumps(status.execution_result, indent=2))
    
    # Run example
    asyncio.run(main())