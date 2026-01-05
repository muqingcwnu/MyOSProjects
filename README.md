# Dandelion-Learn: Learning to Schedule Serverless Computations

This project implements the Dandelion-Learn system for intelligent serverless scheduling using Graph Neural Networks (GNN) and Reinforcement Learning (RL).

## Quick Start

### Run Complete Experiment

To run the entire experiment pipeline (training, evaluation, and figure generation):

```bash
python run_experiment.py
```

This single command will:
1. Run training simulation (2000 iterations, 100 samples each)
2. Run full evaluation (performance, ablation, scalability tests)
3. Generate all figures
4. Save everything to the results folder

**Estimated time: 30-60 minutes**

### Results Structure

After running the experiment, all results will be in a single folder:

```
results/
  ├── *.png             # All figures
  ├── *training*.csv    # Training logs (GNN, RL, Q-values, performance)
  └── *.csv             # Evaluation results
```

## Project Structure

```
.
├── run_experiment.py              # Single entry point to run everything
├── src/
│   └── dandelion_learn/           # Core implementation
│       ├── dandelion_learn_sim.py # Simulator, schedulers, optimizer
│       ├── gnn_predictor.py       # GNN-based performance predictor
│       └── baseline_schedulers.py  # Baseline scheduler implementations
├── scripts/
│   └── real_training_simulation.py    # Training simulation with real-time plots
├── evaluation/
│   ├── simulation_framework.py        # Simulation framework components
│   ├── experimental_evaluation.py     # Experimental evaluation functions
│   ├── generate_figures.py            # Figure generation
│   ├── distributed_evaluation.py      # Distributed scheduling evaluation
│   ├── sensitivity_analysis.py       # Sensitivity analysis
│   └── statistical_tests.py          # Statistical significance tests
├── azure_trace/                    # Azure Functions trace data
└── results/                         # All experiment results (generated)
```

## Key Components

### 1. GNN Predictor
- Predicts function runtime, memory needs, and hardware affinity
- Uses message-passing neural network with 3 layers
- Trained on historical Azure trace data

### 2. RL Scheduler
- Multi-armed bandit approach for hardware selection
- Epsilon-greedy exploration
- Q-value updates based on observed rewards

### 3. Multi-Objective Optimizer
- Balances latency, energy, and cost
- Combines predictions with hardware characteristics
- Fast optimization for real-time scheduling

### 4. Baseline Schedulers
- FIFO, Random, Round-Robin, Locality-Aware, Shortest-Job-First
- Sinan, Fifer, X-FaaS, FIRM (learned schedulers)

## Configuration

### Training Parameters
- Training iterations: 2000
- Samples per iteration: 100
- GNN fine-tuning: Every 20 episodes

### Evaluation Parameters
- Days: 1, 2, 3 (Azure trace days)
- Job counts: 500, 1000, 2000
- Runs per config: 5 (for statistical significance)
- Schedulers: 6 (Dandelion-Learn + 5 baselines)

## Results

After running the experiment, you'll find everything in `results/`:

### Figures (PNG files)
- `figure1_comprehensive_latency.png` - Latency comparisons across schedulers
- `figure2_performance_tradeoffs.png` - Trade-offs between latency, throughput, energy, cost
- `figure3_training_analysis.png` - Training progress and learning curves
- `gnn_detailed_analysis.png` - GNN performance analysis
- Other training and analysis plots

### Data Files (CSV files)
- `performance_results.csv` - Complete evaluation results
- `scalability_overhead.csv` - Scalability analysis
- `ablation_studies.csv` - Component contribution analysis
- `statistical_significance_tests.csv` - Statistical test results

### Training Logs (CSV files)
- `gnn_training.csv` - GNN training metrics over time
- `rl_training.csv` - RL scheduler training metrics
- `rl_q_values.csv` - Q-value evolution during training
- `performance_metrics.csv` - Performance during training

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SimPy
- SciPy

Install dependencies:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install numpy pandas matplotlib seaborn simpy scipy
```
