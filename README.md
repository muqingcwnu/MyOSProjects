# Dandelion-Learn

Learning-enhanced serverless scheduler: GNN predictor + contextual RL + multi-objective optimizer on a SimPy cluster simulator (Azure Functions 2019 traces).

## Setup

```bash
pip install -r requirements.txt
# If azure_trace/*.csv are Git LFS pointers:
#   git lfs pull
```

## Run

```bash
# Quick sanity check
python scripts/smoke_test_pipeline.py

# Full train + eval (writes CSVs to results_fixed/)
python run_experiment.py

# Eval only
python run_complete_evaluation.py

# Server helpers
python train_on_server.py              # smoke
python train_on_server.py --eval-only
python train_on_server.py --full

# Summarize current results
python scripts/summarize_results.py
```

## Layout

```
src/dandelion_learn/     # simulator, GNN, RL, baselines, config
evaluation/              # performance / ablation / scalability / security
scripts/                 # training + smoke + summarize
azure_trace/             # Azure Functions CSVs
results_fixed/           # canonical evaluation + training CSVs
```

## Results (`results_fixed/`)

Canonical evaluation outputs (CSV only; regenerate plots locally if needed):

| File | Contents |
|------|----------|
| `performance_results.csv` | Main scheduler comparison |
| `ablation_studies.csv` | GNN / RL / optimizer ablation |
| `scalability_overhead.csv` | Workload size vs latency / overhead |
| `distributed_scalability.csv` | Multi-node coordination metrics |
| `optimizer_weight_sensitivity.csv` | Optimizer $(\alpha,\beta,\gamma)$ sweep |
| `rl_exploration_sensitivity.csv` | RL $\epsilon$ sweep |
| `statistical_significance_tests.csv` | Welch $t$-tests on p99 |
| `security_analysis.csv` | Modeled CHERI overhead |
| `gnn_training.csv` | GNN loss / errors |
| `rl_training.csv` | RL rewards |
| `rl_q_values.csv` | Q-value trajectories |
| `performance_metrics.csv` | Per-episode training metrics |

Papers / LaTeX sources are not included in this repository.

## Requirements

Python 3.8+, NumPy, Pandas, SimPy, SciPy, PyTorch (for GNN).
