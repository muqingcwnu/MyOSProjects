"""
Distributed coordination protocol for multi-node Dandelion-Learn scheduling.

Implements a lightweight parameter server for sharing Q-values and GNN parameters
across multiple nodes, enabling distributed scheduling with minimal overhead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import threading
import time


@dataclass
class NodeState:
    """State of a single Dandelion-Learn node."""
    node_id: int
    q_values: Dict[int, float]  # hw_id -> Q-value
    last_update_time: float
    num_updates: int


class ParameterServer:
    """
    Lightweight parameter server for distributed RL coordination.
    
    Maintains a shared Q-table that is updated by multiple nodes
    using asynchronous updates with conflict resolution.
    """
    
    def __init__(self, num_hw_units: int, sync_interval: float = 0.1):
        """
        Initialize parameter server.
        
        Args:
            num_hw_units: Number of hardware units in the cluster
            sync_interval: Time interval (seconds) between synchronization attempts
        """
        self.num_hw_units = num_hw_units
        self.sync_interval = sync_interval
        self.lock = threading.Lock()
        
        # Shared Q-table: hw_id -> Q-value
        self.shared_q_values: Dict[int, float] = {
            hw_id: np.random.uniform(-0.1, 0.1) for hw_id in range(num_hw_units)
        }
        
        # Node states for tracking updates
        self.node_states: Dict[int, NodeState] = {}
        
        # Coordination overhead metrics
        self.total_sync_operations = 0
        self.total_sync_time = 0.0
        self.total_conflicts = 0
    
    def register_node(self, node_id: int) -> None:
        """Register a new node with the parameter server."""
        with self.lock:
            self.node_states[node_id] = NodeState(
                node_id=node_id,
                q_values=self.shared_q_values.copy(),
                last_update_time=time.time(),
                num_updates=0
            )
    
    def get_q_values(self, node_id: int) -> Dict[int, float]:
        """
        Get current Q-values for a node.
        
        Returns a copy of the shared Q-table.
        """
        with self.lock:
            if node_id in self.node_states:
                self.node_states[node_id].last_update_time = time.time()
            return self.shared_q_values.copy()
    
    def update_q_values(
        self, 
        node_id: int, 
        updates: Dict[int, float],
        learning_rate: float = 0.1
    ) -> None:
        """
        Update shared Q-values with node's local updates.
        
        Uses weighted averaging to resolve conflicts:
        Q_shared = (1 - α) * Q_shared + α * Q_local
        
        Args:
            node_id: ID of the updating node
            updates: Dictionary of hw_id -> new Q-value
            learning_rate: Weight for local updates (default 0.1)
        """
        start_time = time.time()
        
        with self.lock:
            self.total_sync_operations += 1
            
            # Check for conflicts
            if node_id in self.node_states:
                time_since_update = time.time() - self.node_states[node_id].last_update_time
                if time_since_update < self.sync_interval:
                    self.total_conflicts += 1
            
            # Blend local and shared values
            for hw_id, local_q in updates.items():
                if hw_id in self.shared_q_values:
                    shared_q = self.shared_q_values[hw_id]
                    # Weighted average: favor stable shared, add recent local
                    self.shared_q_values[hw_id] = (
                        (1 - learning_rate) * shared_q + learning_rate * local_q
                    )
                else:
                    self.shared_q_values[hw_id] = local_q
            
            # Update node
            if node_id in self.node_states:
                self.node_states[node_id].q_values = self.shared_q_values.copy()
                self.node_states[node_id].num_updates += 1
                self.node_states[node_id].last_update_time = time.time()
        
        sync_time = time.time() - start_time
        self.total_sync_time += sync_time
    
    def get_coordination_overhead(self) -> Dict[str, float]:
        """
        Get coordination overhead metrics.
        
        Returns:
            Dictionary with overhead statistics:
            - avg_sync_time: Average time per synchronization (ms)
            - total_sync_operations: Total number of sync operations
            - conflict_rate: Percentage of operations with conflicts
            - total_overhead_time: Total time spent on coordination (ms)
        """
        with self.lock:
            avg_sync_time = (
                (self.total_sync_time / self.total_sync_operations * 1000)
                if self.total_sync_operations > 0 else 0.0
            )
            conflict_rate = (
                (self.total_conflicts / self.total_sync_operations * 100)
                if self.total_sync_operations > 0 else 0.0
            )
            
            return {
                'avg_sync_time_ms': avg_sync_time,
                'total_sync_operations': self.total_sync_operations,
                'conflict_rate_percent': conflict_rate,
                'total_overhead_time_ms': self.total_sync_time * 1000
            }


class DistributedRLScheduler:
    """
    RL scheduler with distributed coordination support.
    
    Extends the base RLScheduler to work with a parameter server
    for multi-node coordination.
    """
    
    def __init__(
        self,
        hw_units: List,
        parameter_server: ParameterServer,
        node_id: int,
        epsilon: float = 0.1,
        sync_interval: float = 0.1
    ):
        """
        Initialize distributed RL scheduler.
        
        Args:
            hw_units: List of hardware units
            parameter_server: Shared parameter server instance
            node_id: Unique identifier for this node
            epsilon: Exploration rate for epsilon-greedy
            sync_interval: Time between synchronization attempts (seconds)
        """
        self.hw_units = hw_units
        self.parameter_server = parameter_server
        self.node_id = node_id
        self.epsilon = epsilon
        self.sync_interval = sync_interval
        
        # Local Q-values synced with server
        self.local_q_values: Dict[int, float] = {}
        self.last_sync_time = time.time()
        
        # Update counts for incremental learning
        self.local_counts: Dict[int, int] = {hw.hw_id: 0 for hw in hw_units}
        
        # Register with server
        self.parameter_server.register_node(node_id)
        self._sync_from_server()
    
    def _sync_from_server(self) -> None:
        """Sync Q-values from server."""
        self.local_q_values = self.parameter_server.get_q_values(self.node_id)
        self.last_sync_time = time.time()
    
    def _sync_to_server(self) -> None:
        """Push Q-value updates to server."""
        if time.time() - self.last_sync_time >= self.sync_interval:
            self.parameter_server.update_q_values(
                self.node_id,
                self.local_q_values,
                learning_rate=0.1
            )
            self._sync_from_server()
    
    def select_hw(self, job, now: float):
        """Select hardware using epsilon-greedy."""
        # Sync periodically
        if time.time() - self.last_sync_time >= self.sync_interval:
            self._sync_to_server()
        
        # Epsilon-greedy: explore or exploit
        if np.random.random() < self.epsilon:
            # Random exploration
            return np.random.choice(self.hw_units)
        else:
            # Exploit: pick best Q-value
            best_hw = max(
                self.hw_units,
                key=lambda hw: self.local_q_values.get(hw.hw_id, 0.0)
            )
            return best_hw
    
    def update(self, hw, job, completion_time: float, arrival_time: float) -> None:
        """Update Q-values from reward."""
        latency = completion_time - arrival_time
        reward = -latency  # Negative latency as reward
        
        hw_id = hw.hw_id
        n = self.local_counts[hw_id] + 1
        self.local_counts[hw_id] = n
        
        # Q-learning update
        current_q = self.local_q_values.get(hw_id, 0.0)
        self.local_q_values[hw_id] = current_q + (reward - current_q) / n
        
        # Push to server periodically
        if n % 10 == 0:  # Push every 10 updates
            self._sync_to_server()

