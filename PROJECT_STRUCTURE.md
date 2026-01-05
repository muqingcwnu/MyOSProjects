# Dandelion-Learn Project Structure

## What is This Project?

This project implements Dandelion-Learn, a smart scheduler for serverless computing. Instead of using simple rules, it uses machine learning (Graph Neural Networks and Reinforcement Learning) to make better scheduling decisions. Think of it as teaching a computer to schedule tasks more efficiently by learning from experience.

## Getting Started

To run everything, just use this one command:

```bash
python run_experiment.py
```

That's it! This will run the training, evaluate different schedulers, and generate all the figures. It takes about 30-60 minutes depending on your computer.

## How the Project is Organized

Let me walk you through the main parts of the codebase:

### Core Code (`src/dandelion_learn/`)

This is where the main logic lives:

- **`dandelion_learn_sim.py`**: This is the heart of the system. It contains:
  - The simulator that runs the actual experiments
  - Job definitions (what tasks look like)
  - Hardware units (the machines that run tasks)
  - The schedulers (how we decide where to run tasks)
  - The optimizer (how we balance different goals like speed vs cost)

- **`gnn_predictor.py`**: This is our machine learning component that predicts how long tasks will take, how much memory they need, and which hardware they work best on. It uses a Graph Neural Network, which is good at understanding relationships between different parts of a problem.

- **`baseline_schedulers.py`**: These are the "old school" schedulers we compare against. Things like FIFO (first in, first out), Random, Round-Robin, and some more advanced ones like Shortest-Job-First. We need these to show that our learning-based approach is actually better.

- **`synthetic_dag_workloads.py`**: Sometimes we need to test with controlled workloads, so this generates synthetic task dependencies (like a chain of tasks, or a diamond pattern).

- **`distributed_coordinator.py`**: For when we want to run across multiple machines, this handles the coordination between them.

### Training Scripts (`scripts/`)

- **`real_training_simulation.py`**: This is where the magic happens. It:
  - Loads real Azure trace data (actual serverless workload data)
  - Trains the GNN predictor to learn patterns
  - Trains the RL scheduler to make good decisions
  - Shows you real-time plots as it trains
  - Saves everything to the results folder

### Evaluation Code (`evaluation/`)

This folder contains everything related to testing and comparing schedulers:

- **`simulation_framework.py`**: The building blocks for running experiments - the simulator setup, how to load workloads, which schedulers to test, and what metrics to collect.

- **`experimental_evaluation.py`**: The actual experiments. This runs different schedulers on different workloads and collects performance data. It handles:
  - Performance comparisons
  - Ablation studies (what happens if we remove parts?)
  - Scalability tests (how does it work with more tasks?)
  - Security analysis

- **`generate_figures.py`**: Takes all the data and makes nice figures. Creates multi-panel plots showing latency comparisons, trade-off analyses, training curves, etc.

- **`distributed_evaluation.py`**: Tests how well the system works when spread across multiple machines.

- **`sensitivity_analysis.py`**: Checks how sensitive the results are to different parameter settings (like how much weight we put on latency vs energy).

- **`statistical_tests.py`**: Does proper statistical tests to make sure our improvements are real and not just random chance.

### Main Entry Points

- **`run_experiment.py`**: This is your main command. It runs everything in the right order - training, evaluation, figure generation. Just run this and you're done.

- **`run_complete_evaluation.py`**: If you've already trained and just want to run more evaluations (maybe with different settings), use this.

## How Everything Works Together

Here's the flow of what happens when you run the experiment:

1. **Training Phase**: 
   - We load real Azure trace data (actual serverless function invocations from Microsoft)
   - The GNN predictor learns patterns from historical data
   - The RL scheduler learns by trying different scheduling decisions and seeing what works
   - As it trains, you see real-time plots updating
   - All training data gets saved to `results/`

2. **Evaluation Phase**:
   - We run simulations with different schedulers (ours plus all the baselines)
   - We test on different workloads (different days, different numbers of tasks)
   - We collect metrics like latency, throughput, energy, cost
   - Everything gets saved to `results/performance_results.csv`

3. **Figure Generation**:
   - We read all the evaluation results
   - We create high-quality figures (300 DPI, proper formatting)
   - These go straight into `results/`

## What You Get in the Results Folder

After running, you'll find everything in `results/`:

**Figures (PNG files):**
- `figure1_comprehensive_latency.png` - Shows latency comparisons across schedulers
- `figure2_performance_tradeoffs.png` - Shows trade-offs between latency, throughput, energy, cost
- `figure3_training_analysis.png` - Shows how the learning progressed
- `training_comprehensive.png` - Detailed training plots
- `gnn_detailed_analysis.png` - Deep dive into the GNN performance

**Data Files (CSV files):**
- `performance_results.csv` - All the evaluation results
- `scalability_overhead.csv` - How performance changes with workload size
- `gnn_training.csv` - GNN training metrics over time
- `rl_training.csv` - RL scheduler training metrics
- `rl_q_values.csv` - How the Q-values evolved (RL learning signal)
- `performance_metrics.csv` - Performance during training

## Design Philosophy

We tried to make this project:

1. **Simple to use**: One command does everything
2. **Easy to understand**: Code is clean, well-organized, with clear names
3. **Modular**: Each piece does one thing well and can be used independently
4. **High quality output**: All figures are high resolution and properly formatted
5. **Based on real data**: We use actual Azure traces, not made-up data
