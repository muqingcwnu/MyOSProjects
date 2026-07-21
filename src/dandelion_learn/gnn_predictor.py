from __future__ import annotations

"""
Trainable GNN performance predictor (PyTorch MPNN).

Predicts per-function runtime (ms), memory (MB), and accelerator affinity
from a function-level graph built from the workload.

Training labels (simulation ground truth):
  - runtime: InvocationJob.base_duration
  - memory:  input_size * 1.5  (proxy working-set)
  - gpu affinity: deterministic function of func_id / input size

Cold-start: unseen functions use feature-only MLP embedding (no neighbor msgs).
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

if TYPE_CHECKING:
    from dandelion_learn.dandelion_learn_sim import InvocationJob


@dataclass
class Prediction:
    runtime_ms: float
    mem_mb: float
    gpu_affinity: float


@dataclass
class TrainHistory:
    epochs: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    runtime_mae: List[float] = field(default_factory=list)
    memory_mae: List[float] = field(default_factory=list)


def ground_truth_memory_mb(job: "InvocationJob") -> float:
    return float(max(16.0, job.input_size * 1.5 + 32.0))


def ground_truth_gpu_affinity(job: "InvocationJob") -> float:
    """Deterministic affinity in [0, 1] used as training label and for sim speedup."""
    h = abs(hash(job.func_id)) % 10_000
    size_factor = float(np.tanh(job.input_size / 10.0))
    return float(0.15 + 0.7 * (h / 10_000.0) + 0.15 * size_factor)


def _node_features(job: "InvocationJob", d_node: int = 8) -> np.ndarray:
    x = np.zeros(d_node, dtype=np.float32)
    code_hash = (abs(hash(job.func_id)) % (10**6)) / float(10**6)
    x[0] = code_hash
    x[1] = float(np.log1p(job.input_size))
    x[2] = float(np.log1p(job.base_duration))
    x[3] = 1.0  # cpu preference prior
    x[4] = ground_truth_gpu_affinity(job)
    x[5] = float(job.input_size / (job.input_size + 10.0))
    return x


if HAS_TORCH:

    class _MPNN(nn.Module):
        def __init__(self, d_node: int, d_hidden: int, d_edge: int, L: int) -> None:
            super().__init__()
            self.L = L
            self.emb = nn.Linear(d_node, d_hidden)
            self.edge_mlp = nn.Linear(d_edge, d_hidden)
            self.layers = nn.ModuleList(
                [nn.Linear(2 * d_hidden, d_hidden) for _ in range(L)]
            )
            self.head_t = nn.Linear(d_hidden, 1)
            self.head_m = nn.Linear(d_hidden, 1)
            self.head_g = nn.Linear(d_hidden, 1)

        def forward(
            self,
            x: "torch.Tensor",
            edge_index: "torch.Tensor",
            edge_attr: "torch.Tensor",
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            h = F.relu(self.emb(x))
            if edge_index.numel() == 0:
                t = self.head_t(h).squeeze(-1)
                m = F.relu(self.head_m(h).squeeze(-1))
                g = torch.sigmoid(self.head_g(h).squeeze(-1))
                return t, m, g

            src, dst = edge_index[0], edge_index[1]
            for layer in self.layers:
                phi_e = self.edge_mlp(edge_attr)
                msg = h[src] * phi_e
                agg = torch.zeros_like(h)
                agg.index_add_(0, dst, msg)
                h = F.relu(layer(torch.cat([h, agg], dim=-1)))

            t = self.head_t(h).squeeze(-1)
            m = F.relu(self.head_m(h).squeeze(-1))
            g = torch.sigmoid(self.head_g(h).squeeze(-1))
            return t, m, g


class GNNPredictor:
    """Trainable GNN predictor with cold-start fallback."""

    def __init__(
        self,
        d_node: int = 8,
        d_hidden: int = 64,
        L: int = 3,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for GNNPredictor. Install with: pip install torch"
            )

        from dandelion_learn.experiment_config import resolve_device

        self.d_node = d_node
        self.d_hidden = d_hidden
        self.L = L
        self.d_edge = 4
        resolved = resolve_device(device)
        self.device = torch.device(resolved)

        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        self.model = _MPNN(d_node, d_hidden, self.d_edge, L)
        try:
            self.model = self.model.to(self.device)
        except RuntimeError as exc:
            print(f"[WARN] Moving GNN to {self.device} failed ({exc}); using CPU.")
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)
        self.func_ids: List[str] = []
        self.func_index: Dict[str, int] = {}
        self._func_centroid: Dict[str, np.ndarray] = {}
        self._global_mean_runtime = 5.0
        self._global_mean_mem = 48.0
        self._trained = False
        self.history = TrainHistory()

    def _build_graph(
        self, jobs: Sequence["InvocationJob"]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        by_func: Dict[str, List["InvocationJob"]] = {}
        for j in jobs:
            by_func.setdefault(j.func_id, []).append(j)

        self.func_ids = sorted(by_func.keys())
        self.func_index = {fid: i for i, fid in enumerate(self.func_ids)}
        n = len(self.func_ids)

        X = np.zeros((n, self.d_node), dtype=np.float32)
        y_t = np.zeros(n, dtype=np.float32)
        y_m = np.zeros(n, dtype=np.float32)
        y_g = np.zeros(n, dtype=np.float32)

        for fid, idx in self.func_index.items():
            js = by_func[fid]
            # Aggregate features / labels per function
            feats = np.stack([_node_features(j, self.d_node) for j in js], axis=0)
            X[idx] = feats.mean(axis=0)
            y_t[idx] = float(np.mean([j.base_duration for j in js]))
            y_m[idx] = float(np.mean([ground_truth_memory_mb(j) for j in js]))
            y_g[idx] = float(np.mean([ground_truth_gpu_affinity(j) for j in js]))
            self._func_centroid[fid] = X[idx].copy()

        self._global_mean_runtime = float(np.mean(y_t)) if n else 5.0
        self._global_mean_mem = float(np.mean(y_m)) if n else 48.0

        # Edges: kNN on log-input feature
        edges: List[Tuple[int, int]] = []
        mean_inputs = X[:, 1]
        k = min(3, max(0, n - 1))
        for i in range(n):
            if k == 0:
                break
            dists = np.abs(mean_inputs - mean_inputs[i])
            neigh = np.argsort(dists)[1 : k + 1]
            for j in neigh:
                edges.append((i, int(j)))
                edges.append((int(j), i))

        if edges:
            edge_index = np.array(edges, dtype=np.int64).T
            E = np.zeros((len(edges), self.d_edge), dtype=np.float32)
            for k_e, (i, j) in enumerate(edges):
                data_size = float(abs(mean_inputs[i] - mean_inputs[j]) + 1.0)
                E[k_e, 0] = np.log1p(data_size)
                E[k_e, 1] = 0.1
                E[k_e, 2] = float(abs(y_t[i] - y_t[j]))
                E[k_e, 3] = float(abs(y_g[i] - y_g[j]))
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            E = np.zeros((0, self.d_edge), dtype=np.float32)

        return X, edge_index, E, y_t, y_m, y_g

    def fit(
        self,
        jobs: List["InvocationJob"],
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 256,
        weight_decay: float = 1e-5,
        verbose: bool = False,
    ) -> TrainHistory:
        """
        Train the GNN on labeled jobs (offline pre-train).

        Uses full-graph message passing with mini-batch loss over nodes
        when the function graph is larger than ``batch_size``.
        """
        if not jobs:
            return self.history

        X, edge_index, E, y_t, y_m, y_g = self._build_graph(jobs)
        n = X.shape[0]
        x_t = torch.tensor(X, device=self.device)
        ei_t = torch.tensor(edge_index, device=self.device)
        e_t = torch.tensor(E, device=self.device)
        yt = torch.tensor(y_t, device=self.device)
        ym = torch.tensor(y_m, device=self.device)
        yg = torch.tensor(y_g, device=self.device)

        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.history = TrainHistory()
        self.model.train()

        yt_log = torch.log1p(yt)
        ym_log = torch.log1p(ym)
        bs = max(1, int(batch_size))

        if verbose:
            print(
                f"  GNN train: nodes={n} edges={edge_index.shape[1]} "
                f"epochs={epochs} batch_size={bs} device={self.device}"
            )

        for epoch in range(1, epochs + 1):
            perm = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            epoch_mae_t = 0.0
            epoch_mae_m = 0.0
            n_batches = 0

            for start in range(0, n, bs):
                idx = perm[start : start + bs]
                opt.zero_grad()
                pred_t, pred_m, pred_g = self.model(x_t, ei_t, e_t)
                loss_t = F.mse_loss(pred_t[idx], yt_log[idx])
                loss_m = F.mse_loss(pred_m[idx], ym_log[idx])
                loss_g = F.mse_loss(pred_g[idx], yg[idx])
                loss = loss_t + loss_m + 0.5 * loss_g
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()

                with torch.no_grad():
                    mae_t = torch.mean(torch.abs(torch.expm1(pred_t[idx]) - yt[idx]))
                    mae_m = torch.mean(torch.abs(torch.expm1(pred_m[idx]) - ym[idx]))
                epoch_loss += float(loss.item())
                epoch_mae_t += float(mae_t.item())
                epoch_mae_m += float(mae_m.item())
                n_batches += 1

            epoch_loss /= max(1, n_batches)
            epoch_mae_t /= max(1, n_batches)
            epoch_mae_m /= max(1, n_batches)
            self.history.epochs.append(epoch)
            self.history.losses.append(epoch_loss)
            self.history.runtime_mae.append(epoch_mae_t)
            self.history.memory_mae.append(epoch_mae_m)

            if verbose and (epoch == 1 or epoch == epochs or epoch % max(1, epochs // 5) == 0):
                print(
                    f"  GNN epoch {epoch}/{epochs}: loss={epoch_loss:.4f} "
                    f"runtime_MAE={epoch_mae_t:.3f} mem_MAE={epoch_mae_m:.3f}"
                )

        self._trained = True
        self.model.eval()
        return self.history

    def _predict_features(self, x: np.ndarray) -> Prediction:
        """Cold-start / single-node forward (no graph neighbors)."""
        self.model.eval()
        with torch.no_grad():
            xt = torch.tensor(x[None, :], device=self.device)
            ei = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            ea = torch.zeros((0, self.d_edge), device=self.device)
            t, m, g = self.model(xt, ei, ea)
        # Heads trained in log1p space
        runtime = max(0.5, float(np.expm1(t.item())))
        mem = max(16.0, float(np.expm1(m.item())))
        aff = float(np.clip(g.item(), 0.0, 1.0))
        return Prediction(runtime_ms=runtime, mem_mb=mem, gpu_affinity=aff)

    def predict(self, job: "InvocationJob", hw_type: str = "cpu") -> Prediction:
        x = _node_features(job, self.d_node)

        if self._trained and job.func_id in self.func_index:
            # Rebuild mini-graph around known functions for a consistent embedding
            # Fast path: use cached centroid + single-node head (trained weights)
            if job.func_id in self._func_centroid:
                x = 0.7 * self._func_centroid[job.func_id] + 0.3 * x

        pred = self._predict_features(x)

        # Mild hardware-conditioned affinity adjustment (prediction only)
        gpu_affinity = pred.gpu_affinity
        if hw_type == "gpu":
            gpu_affinity = min(1.0, gpu_affinity + 0.05)
        elif hw_type == "fpga":
            gpu_affinity = min(1.0, 0.5 * gpu_affinity + 0.25)
        elif hw_type == "cpu":
            gpu_affinity = max(0.0, gpu_affinity - 0.05)

        # Cold-start shrinkage toward global means when never trained / unseen
        if not self._trained or job.func_id not in self.func_index:
            runtime = 0.5 * pred.runtime_ms + 0.5 * self._global_mean_runtime
            mem = 0.5 * pred.mem_mb + 0.5 * self._global_mean_mem
            return Prediction(runtime_ms=max(0.5, runtime), mem_mb=max(16.0, mem), gpu_affinity=gpu_affinity)

        return Prediction(
            runtime_ms=pred.runtime_ms,
            mem_mb=pred.mem_mb,
            gpu_affinity=gpu_affinity,
        )

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "func_centroid": self._func_centroid,
                "func_index": self.func_index,
                "func_ids": self.func_ids,
                "global_mean_runtime": self._global_mean_runtime,
                "global_mean_mem": self._global_mean_mem,
                "d_node": self.d_node,
                "d_hidden": self.d_hidden,
                "L": self.L,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self._func_centroid = ckpt.get("func_centroid", {})
        self.func_index = ckpt.get("func_index", {})
        self.func_ids = ckpt.get("func_ids", [])
        self._global_mean_runtime = ckpt.get("global_mean_runtime", 5.0)
        self._global_mean_mem = ckpt.get("global_mean_mem", 48.0)
        self._trained = True
        self.model.eval()
