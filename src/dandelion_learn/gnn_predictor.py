from __future__ import annotations

"""
GNN-style performance predictor implemented in a lightweight NumPy-only way.

We implement:
  - Node features x_i
  - Edge features e_ij
  - A 3-layer message passing network:

      h_i^{(0)} = MLP_emb(x_i)
      h_i^{(l+1)} = sigma(
          W^{(l)} · CONCAT(
              h_i^{(l)},
              sum_{j in N(i)} h_j^{(l)} ⊙ phi(e_ij)
          ) + b^{(l)}
      )

  - Prediction heads:

      t_hat_i = W_t h_i^{(L)} + b_t
      m_hat_i = ReLU(W_m h_i^{(L)} + b_m)
      g_hat_i^(·) = sigmoid(W_g^(·) h_i^{(L)} + b_g^(·))

For simplicity we:
  - Build a function-level graph from the workload (functions are nodes).
  - Connect functions with synthetic edges based on similarity of their
    approximate input sizes (to provide some neighborhood structure).
  - Initialize weights randomly and do not train; the goal is to follow
    the structure of the equations, not achieve production-quality predictions.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from dandelion_learn.dandelion_learn_sim import InvocationJob


@dataclass
class Prediction:
    runtime_ms: float
    mem_mb: float
    gpu_affinity: float


class GNNPredictor:
    def __init__(self, d_node: int = 8, d_hidden: int = 16, L: int = 3) -> None:
        self.d_node = d_node
        self.d_hidden = d_hidden
        self.L = L

        # Node embedding (single linear layer)
        self.W_emb = self._randn((d_hidden, d_node))
        self.b_emb = self._randn((d_hidden,))

        # Message passing layers
        self.W_layers: List[np.ndarray] = []
        self.b_layers: List[np.ndarray] = []
        # Layer input is concat(h_i, m_i), both d_hidden
        d_in = 2 * d_hidden
        for _ in range(L):
            self.W_layers.append(self._randn((d_hidden, d_in)))
            self.b_layers.append(self._randn((d_hidden,)))

        # Edge feature projection: d_edge -> d_hidden
        self.d_edge = 4
        self.W_edge = self._randn((d_hidden, self.d_edge))

        # Prediction heads
        self.W_t = self._randn((1, d_hidden))
        self.b_t = self._randn((1,))

        self.W_m = self._randn((1, d_hidden))
        self.b_m = self._randn((1,))

        self.W_g = self._randn((1, d_hidden))
        self.b_g = self._randn((1,))

        # Cache function embeddings
        self.func_ids: List[str] = []
        self.func_index: Dict[str, int] = {}
        self.h_L: np.ndarray | None = None

    @staticmethod
    def _randn(shape) -> np.ndarray:
        return np.random.randn(*shape).astype(np.float32) * 0.1

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, jobs: List["InvocationJob"]) -> None:
        """
        Build a function-level graph and run the GNN forward pass once.
        """
        # Group jobs by function
        by_func: Dict[str, List["InvocationJob"]] = {}
        for j in jobs:
            by_func.setdefault(j.func_id, []).append(j)

        self.func_ids = sorted(by_func.keys())
        self.func_index = {fid: i for i, fid in enumerate(self.func_ids)}
        n = len(self.func_ids)

        # Node features: code hash, log input size, log duration, hw one-hot
        X = np.zeros((n, self.d_node), dtype=np.float32)
        for fid, idx in self.func_index.items():
            js = by_func[fid]
            mean_input = float(np.mean([j.input_size for j in js]))
            mean_dur = float(np.mean([j.base_duration for j in js]))
            code_hash = (hash(fid) % (10**6)) / (10**6)

            # Feature layout: [hash, log_input, log_duration, cpu=1, gpu=0, ...]
            X[idx, 0] = code_hash
            X[idx, 1] = np.log1p(mean_input)
            X[idx, 2] = np.log1p(mean_dur)
            X[idx, 3] = 1.0
            X[idx, 4] = 0.0

        # Connect functions with similar input sizes
        edges: List[Tuple[int, int]] = []
        mean_inputs = np.array([X[i, 1] for i in range(n)])
        for i in range(n):
            # Connect to 3 nearest neighbors
            dists = np.abs(mean_inputs - mean_inputs[i])
            neigh_idx = np.argsort(dists)[:4]  # self + 3 nearest
            for j in neigh_idx:
                if i == j:
                    continue
                edges.append((i, j))

        # Edge features: log data size, network latency, padding
        E = np.zeros((len(edges), self.d_edge), dtype=np.float32)
        for k, (i, j) in enumerate(edges):
            # Use input size diff as proxy for data size
            data_size = float(abs(mean_inputs[i] - mean_inputs[j]) + 1.0)
            E[k, 0] = np.log1p(data_size)
            E[k, 1] = 0.1  # synthetic network latency

        # Build adjacency list
        neighbors: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(n)}
        for eid, (i, j) in enumerate(edges):
            neighbors[i].append((j, eid))

        # Message passing forward pass
        h = self._relu((X @ self.W_emb.T) + self.b_emb)  # (n, d_hidden)

        for l in range(self.L):
            # Aggregate neighbor messages
            m = np.zeros_like(h)
            for i in range(n):
                if not neighbors[i]:
                    continue
                acc = np.zeros((self.d_hidden,), dtype=np.float32)
                for j, eid in neighbors[i]:
                    h_j = h[j]
                    e_ij = E[eid]
                    phi_e = self.W_edge @ e_ij  # (d_hidden,)
                    acc += h_j * phi_e
                m[i] = acc

            # Update: concat then linear transform
            concat = np.concatenate([h, m], axis=1)  # (n, 2*d_hidden)
            W = self.W_layers[l]
            b = self.b_layers[l]
            h = self._relu((concat @ W.T) + b)

        self.h_L = h

    def _get_embedding(self, func_id: str) -> np.ndarray:
        if self.h_L is None or func_id not in self.func_index:
            # Random embedding for unknown functions
            return self._randn((self.d_hidden,))
        return self.h_L[self.func_index[func_id]]

    def predict(self, job: "InvocationJob", hw_type: str = "cpu") -> Prediction:
        """
        Predict runtime, memory, and GPU affinity for a specific job using
        the node embedding h_L of its function.
        """
        h_i = self._get_embedding(job.func_id)

        # Compute predictions
        t_hat = float(self.W_t @ h_i + self.b_t)  # runtime
        m_hat = float(self._relu(self.W_m @ h_i + self.b_m))  # memory
        g_hat = float(self._sigmoid(self.W_g @ h_i + self.b_g))  # generic affinity

        # Normalize outputs and adjust for hw type
        runtime_ms = max(0.5, abs(t_hat) + 5.0)  # keep > 0
        mem_mb = max(16.0, abs(m_hat) * 4.0 + 32.0)
        gpu_affinity = g_hat
        if hw_type == "gpu":
            gpu_affinity = min(1.0, gpu_affinity + 0.2)

        return Prediction(runtime_ms=runtime_ms, mem_mb=mem_mb, gpu_affinity=gpu_affinity)



