"""
Dandelion-Learn Complete Experiment Runner

Main entry point to run the full experiment pipeline:
- Training simulation
- Performance evaluation
- Figure generation

Usage:
    python run_experiment.py
"""

import sys
from pathlib import Path
import subprocess
import shutil
import time

# Setup paths
root = Path(__file__).parent.resolve()
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "src"))

print("Dandelion-Learn Experiment Runner")
print("\nThis script will:")
print("  1. Run training simulation (2000 iterations, 100 samples each)")
print("  2. Run experimental evaluation (performance, ablation, scalability)")
print("  3. Generate all figures")
print("\nEstimated time: 30-60 minutes")

# Setup results directory
results_dir = root / "results"
results_dir.mkdir(parents=True, exist_ok=True)

print(f"\n[OK] Created results directory: {results_dir}")

# Step 1: Training
print("\nStep 1: Training Simulation")

start_time = time.time()

try:
    result = subprocess.run(
        [sys.executable, "scripts/real_training_simulation.py"],
        capture_output=True,
        text=True,
        check=True
    )
    print(result.stdout)
    
    # Logs saved to results
    print(f"\n[OK] Training logs and plots saved to: {results_dir}")
    
    elapsed = time.time() - start_time
    print(f"\n[OK] Training simulation completed in {elapsed/60:.1f} minutes")
    
except subprocess.CalledProcessError as e:
    print(f"\n[ERROR] Training simulation failed:")
    print(e.stderr)
    sys.exit(1)
except Exception as e:
    print(f"\n[ERROR] Error: {e}")
    sys.exit(1)

# Step 2: Evaluation
print("\nStep 2: Full Evaluation")

start_time = time.time()

try:
    # Run evaluation
    result = subprocess.run(
        [sys.executable, "run_complete_evaluation.py"],
        capture_output=True,
        text=True,
        check=True
    )
    print(result.stdout)
    
    elapsed = time.time() - start_time
    print(f"\n[OK] Full evaluation completed in {elapsed/60:.1f} minutes")
    
except subprocess.CalledProcessError as e:
    print(f"\n[ERROR] Full evaluation failed:")
    print(e.stderr)
    sys.exit(1)
except Exception as e:
    print(f"\n[ERROR] Error: {e}")
    sys.exit(1)

# Step 3: Generate figures
print("\nStep 3: Generating Figures")
try:
    result = subprocess.run(
        [sys.executable, "evaluation/generate_figures.py"],
        capture_output=True,
        text=True,
        check=True
    )
    print(result.stdout)
    
        
except Exception as e:
    print(f"[WARNING] Evaluation figures generation had issues: {e}")

# Step 4: Summary
print("\nStep 4: Summary")

# Count files
all_figures = sorted(results_dir.glob("*.png"))
all_logs = sorted(results_dir.glob("*training*.csv"))
all_data = sorted(results_dir.glob("*.csv"))
all_data = [f for f in all_data if "training" not in f.name.lower()]

print(f"\n[OK] Total figures generated: {len(all_figures)}")
print(f"[OK] Total log files: {len(all_logs)}")
print(f"[OK] Total data files: {len(all_data)}")

print(f"\nResults summary:")
print(f"  - Figures: {len(all_figures)} PNG files")
print(f"  - Logs:   {len(all_logs)} CSV files")
print(f"  - Data:   {len(all_data)} CSV files")

print("\n[OK] Experiment complete!")
print(f"\nAll results are in: {results_dir}")

