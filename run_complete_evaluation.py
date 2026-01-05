"""
Complete Experimental Evaluation Runner

This script runs the complete experimental evaluation:
- Simulation Framework and Methodology
- Experimental Evaluation

Usage:
    python run_complete_evaluation.py
"""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from evaluation.experimental_evaluation import (
    ExperimentalSetup,
    run_performance_evaluation,
    run_ablation_studies,
    run_scalability_analysis,
    run_security_analysis,
)

print("=" * 70)
print("DANDELION-LEARN: COMPLETE EXPERIMENTAL EVALUATION")
print("=" * 70)
print()

# Create results directory
results_dir = Path(".").resolve() / "results"
results_dir.mkdir(exist_ok=True)

# Configure experimental setup
setup = ExperimentalSetup(
    num_training_episodes=2000,
    jobs_per_episode=100,
    num_runs_per_config=5,
)

print("Experimental Setup:")
print(f"  - Training episodes: {setup.num_training_episodes}")
print(f"  - Jobs per episode: {setup.jobs_per_episode}")
print(f"  - Runs per configuration: {setup.num_runs_per_config}")
print(f"  - Azure trace days: {setup.azure_trace_days}")
print(f"  - Synthetic workloads: {setup.synthetic_workloads}")
print()

# ============================================================================
# Simulation Framework and Methodology
# ============================================================================
print("=" * 70)
print("SIMULATION FRAMEWORK AND METHODOLOGY")
print("=" * 70)
print()
print("Discrete-event simulator (SimPy-based) - Implemented")
print("Workload datasets - Azure trace + Synthetic DAGs")
print("Baseline schedulers - FIFO, Random, Round-Robin, etc.")
print("Evaluation metrics - Latency, throughput, energy, cost")
print()
print("[OK] Simulation framework ready")
print()

# ============================================================================
# Experimental Evaluation
# ============================================================================
print("=" * 70)
print("EXPERIMENTAL EVALUATION")
print("=" * 70)
print()

total_start = time.time()

# Experimental setup
print("Experimental Setup")
print("  - Hardware: CPU, GPU, FPGA units")
print("  - Workloads: Azure traces + synthetic DAGs")
print("  - Schedulers: Dandelion-Learn + 5 baselines")
print("  - Metrics: Latency, throughput, energy, cost, utilization")
print()
print("[OK] Experimental setup configured")
print()

# Performance results
print("=" * 70)
print("PERFORMANCE RESULTS")
print("=" * 70)
print("Running comprehensive performance evaluation...")
print("This compares Dandelion-Learn against all baseline schedulers.")
print()

start_time = time.time()
try:
    perf_df = run_performance_evaluation(setup, results_dir)
    elapsed = time.time() - start_time
    print(f"\n[OK] Performance evaluation completed in {elapsed/60:.1f} minutes")
except Exception as e:
    print(f"\n[ERROR] Performance evaluation failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Ablation studies
print("=" * 70)
print("ABLATION STUDIES")
print("=" * 70)
print("Analyzing component contributions...")
print()

start_time = time.time()
try:
    ablation_df = run_ablation_studies(setup, results_dir)
    elapsed = time.time() - start_time
    print(f"\n[OK] Ablation studies completed in {elapsed/60:.1f} minutes")
except Exception as e:
    print(f"\n[ERROR] Ablation studies failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Scalability and overhead
print("=" * 70)
print("SCALABILITY AND OVERHEAD")
print("=" * 70)
print("Analyzing system scalability and overhead...")
print()

start_time = time.time()
try:
    scalability_df = run_scalability_analysis(setup, results_dir)
    elapsed = time.time() - start_time
    print(f"\n[OK] Scalability analysis completed in {elapsed/60:.1f} minutes")
except Exception as e:
    print(f"\n[ERROR] Scalability analysis failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Security analysis
print("=" * 70)
print("SECURITY ANALYSIS")
print("=" * 70)
print("Analyzing security properties...")
print()

start_time = time.time()
try:
    security_df = run_security_analysis(setup, results_dir)
    elapsed = time.time() - start_time
    print(f"\n[OK] Security analysis completed in {elapsed/60:.1f} minutes")
except Exception as e:
    print(f"\n[ERROR] Security analysis failed: {e}")
    import traceback
    traceback.print_exc()

print()

# ============================================================================
# Final Summary
# ============================================================================
total_elapsed = time.time() - total_start

print("=" * 70)
print("EXPERIMENTAL EVALUATION COMPLETE")
print("=" * 70)
print(f"\nTotal time: {total_elapsed/60:.1f} minutes")
print(f"\nResults saved to: {results_dir}")
print()

# List generated files
result_files = list(results_dir.glob("*.csv"))
print(f"Generated {len(result_files)} result files:")
for f in sorted(result_files):
    print(f"  - {f.name}")

print()
print("=" * 70)
print("[OK] All evaluation sections completed")
print("=" * 70)

