"""
Real Training Simulation with Real-Time Plot Generation

Runs training simulation and generates plots in real-time, saving them
directly to the results folder.
"""

import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib
# Set backend before importing pyplot - this ensures it works in PyCharm
try:
    matplotlib.use('TkAgg')  # TkAgg works well with PyCharm
except ImportError:
    try:
        matplotlib.use('Qt5Agg')  # Fallback to Qt5Agg
    except ImportError:
        pass  # Use default backend
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Enable interactive mode for real-time plotting
plt.ion()

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dandelion_learn.dandelion_learn_sim import (
    InvocationJob,
    load_azure_invocations,
    build_default_cluster,
    GNNPredictor,
    RLScheduler,
    MultiObjectiveOptimizer,
    ClusterSimulator,
)

# Consistent styling configuration for all plots
PLOT_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
}

plt.rcParams.update(PLOT_STYLE)

COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#28A745',
}


class RealTrainingLogger:
    """Logs actual training metrics for GNN and RL."""
    
    def __init__(self):
        # GNN training metrics
        self.gnn_epochs: List[int] = []
        self.gnn_losses: List[float] = []
        self.gnn_runtime_errors: List[float] = []
        self.gnn_memory_errors: List[float] = []
        
        # RL training metrics
        self.rl_episodes: List[int] = []
        self.rl_rewards: List[float] = []
        self.rl_q_values: Dict[int, List[float]] = {}
        self.rl_latencies: List[float] = []
        
        # Performance metrics
        self.episode_latencies: List[float] = []
        self.episode_throughputs: List[float] = []
        self.episode_gpu_utils: List[float] = []
    
    def log_gnn_training(self, epoch: int, loss: float, runtime_error: float, memory_error: float):
        self.gnn_epochs.append(epoch)
        self.gnn_losses.append(loss)
        self.gnn_runtime_errors.append(runtime_error)
        self.gnn_memory_errors.append(memory_error)
    
    def log_rl_update(self, episode: int, hw_id: int, q_value: float, reward: float, latency: float):
        self.rl_episodes.append(episode)
        self.rl_rewards.append(reward)
        if hw_id not in self.rl_q_values:
            self.rl_q_values[hw_id] = []
        self.rl_q_values[hw_id].append(q_value)
        self.rl_latencies.append(latency)
    
    def log_episode_performance(self, latency: float, throughput: float, gpu_util: float):
        self.episode_latencies.append(latency)
        self.episode_throughputs.append(throughput)
        self.episode_gpu_utils.append(gpu_util)


def compute_prediction_error(predictor: GNNPredictor, jobs: List[InvocationJob], hw_type: str = "cpu") -> tuple[float, float]:
    """Compute prediction error for runtime and memory."""
    runtime_errors = []
    memory_errors = []
    
    for job in jobs[:100]:  # Sample for speed
        pred = predictor.predict(job, hw_type=hw_type)
        # Use base_duration as ground truth approximation
        actual_runtime = job.base_duration
        pred_runtime = pred.runtime_ms
        runtime_errors.append(abs(actual_runtime - pred_runtime) / (actual_runtime + 1e-6))
        
        # Memory error (simplified)
        actual_memory = job.input_size * 1.5
        pred_memory = pred.mem_mb
        memory_errors.append(abs(actual_memory - pred_memory) / (actual_memory + 1e-6))
    
    return np.mean(runtime_errors), np.mean(memory_errors)


class RealTimePlotter:
    """Manages a single real-time plot window with animated updates."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.fig = None
        self.axes = None
        self.lines = {}
        self.initialized = False
    
    def _apply_consistent_style(self, ax, xlabel: str, ylabel: str, title: str, 
                                yscale: str = 'linear', twin_ax=None):
        """Apply consistent styling to an axis."""
        ax.set_xlabel(xlabel, fontweight='bold', fontsize=PLOT_STYLE['axes.labelsize'])
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=PLOT_STYLE['axes.labelsize'])
        ax.set_title(title, fontweight='bold', fontsize=PLOT_STYLE['axes.titlesize'], pad=10)
        ax.set_yscale(yscale)
        ax.grid(alpha=0.3, linestyle='--', linewidth=PLOT_STYLE['grid.linewidth'])
        ax.tick_params(axis='both', labelsize=PLOT_STYLE['xtick.labelsize'], 
                      width=PLOT_STYLE['xtick.major.width'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(PLOT_STYLE['axes.linewidth'])
        ax.spines['bottom'].set_linewidth(PLOT_STYLE['axes.linewidth'])
        
        if twin_ax is not None:
            # Only apply tick and spine styling to twin axis, ylabel is set manually
            twin_ax.tick_params(axis='y', labelsize=PLOT_STYLE['ytick.labelsize'],
                               width=PLOT_STYLE['ytick.major.width'])
            twin_ax.spines['top'].set_visible(False)
            twin_ax.spines['right'].set_visible(False)
    
    def initialize(self):
        """Create the initial figure with 2x2 subplots."""
        if self.initialized:
            return
        
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle("Dandelion-Learn Training Progress (Real-Time)", 
                         fontsize=16, fontweight='bold', y=0.995)
        
        # Subplot 1: GNN Loss
        ax1 = self.axes[0, 0]
        self._apply_consistent_style(ax1, "Training Epoch", "Loss (log scale)", 
                                   "GNN Training Loss", yscale='log')
        self.lines['gnn_loss'] = ax1.plot([], [], 'o-', linewidth=2, color=COLORS['primary'], 
                                          markersize=4, label='Loss', alpha=0.8)[0]
        self.lines['gnn_smooth'] = ax1.plot([], [], '--', linewidth=1.5, color=COLORS['secondary'], 
                                           label='Smoothed', alpha=0.7)[0]
        ax1.legend(loc='upper right', fontsize=PLOT_STYLE['legend.fontsize'], 
                  frameon=True, fancybox=True, shadow=True)
        
        # Subplot 2: RL Reward
        ax2 = self.axes[0, 1]
        self._apply_consistent_style(ax2, "Training Episode", "Reward", 
                                   "RL Reward Convergence")
        self.lines['rl_reward'] = ax2.plot([], [], '-', linewidth=1, color=COLORS['secondary'], 
                                          alpha=0.3, label='Raw', zorder=1)[0]
        self.lines['rl_smooth'] = ax2.plot([], [], '-', linewidth=2, color=COLORS['primary'], 
                                          label='Smoothed', alpha=0.9, zorder=3)[0]
        ax2.legend(loc='lower right', fontsize=PLOT_STYLE['legend.fontsize'],
                  frameon=True, fancybox=True, shadow=True)
        
        # Subplot 3: Q-Values
        ax3 = self.axes[1, 0]
        self._apply_consistent_style(ax3, "Training Episode", "Q-Value", 
                                   "Q-Value Evolution")
        self.lines['q_values'] = {}  # Will store lines for each hardware unit
        
        # Subplot 4: Performance
        ax4 = self.axes[1, 1]
        ax4_twin = ax4.twinx()
        self._apply_consistent_style(ax4, "Training Episode", "Tail Latency (p99, ms)", 
                                   "Performance During Training", twin_ax=ax4_twin)
        ax4.set_ylabel("Tail Latency (p99, ms)", fontweight='bold', 
                      fontsize=PLOT_STYLE['axes.labelsize'], color=COLORS['primary'])
        ax4_twin.set_ylabel("Throughput (jobs/s)", fontweight='bold', 
                          fontsize=PLOT_STYLE['axes.labelsize'], color=COLORS['accent'])
        ax4.grid(alpha=0.3, linestyle='--', linewidth=PLOT_STYLE['grid.linewidth'])
        self.lines['latency'] = ax4.plot([], [], '-', linewidth=2, color=COLORS['primary'], 
                                        label='Tail Latency (p99)', alpha=0.9)[0]
        self.lines['throughput'] = ax4_twin.plot([], [], '-', linewidth=2, color=COLORS['accent'], 
                                                 label='Throughput', alpha=0.9)[0]
        lines_combined = [self.lines['latency'], self.lines['throughput']]
        labels_combined = [l.get_label() for l in lines_combined]
        ax4.legend(lines_combined, labels_combined, loc='upper left', 
                  fontsize=PLOT_STYLE['legend.fontsize'], frameon=True, fancybox=True, shadow=True)
        ax4.tick_params(axis='y', labelcolor=COLORS['primary'], 
                       labelsize=PLOT_STYLE['ytick.labelsize'])
        ax4_twin.tick_params(axis='y', labelcolor=COLORS['accent'],
                            labelsize=PLOT_STYLE['ytick.labelsize'])
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        try:
            plt.show(block=False)
            plt.pause(0.1)
            # Force the window to appear on top
            if hasattr(self.fig.canvas, 'manager'):
                manager = self.fig.canvas.manager
                if hasattr(manager, 'window'):
                    manager.window.wm_attributes('-topmost', 1)
                    manager.window.wm_attributes('-topmost', 0)
        except Exception as e:
            print(f"[WARNING] Could not display interactive plot: {e}")
            print("[INFO] Plots will still be saved to files at the end.")
        
        self.initialized = True
    
    def update(self, logger: RealTrainingLogger, episode: int):
        """Update the plot with latest data."""
        if not self.initialized:
            self.initialize()
        
        # Update GNN Loss
        if len(logger.gnn_epochs) > 0:
            epochs = np.array(logger.gnn_epochs)
            loss = np.array(logger.gnn_losses)
            self.lines['gnn_loss'].set_data(epochs, loss)
            
            if len(loss) > 5:
                smoothed = savgol_filter(loss, min(5, len(loss)//2*2+1), 3)
                self.lines['gnn_smooth'].set_data(epochs, smoothed)
            
            ax1 = self.axes[0, 0]
            ax1.relim()
            ax1.autoscale_view()
        
        # Update RL Reward
        if len(logger.rl_episodes) > 0:
            episodes = np.array(logger.rl_episodes)
            rewards = np.array(logger.rl_rewards)
            self.lines['rl_reward'].set_data(episodes, rewards)
            
            if len(rewards) > 20:
                window = max(1, len(rewards) // 20)
                smoothed = pd.Series(rewards).rolling(window=window, center=True).mean()
                self.lines['rl_smooth'].set_data(episodes, smoothed.values)
            else:
                self.lines['rl_smooth'].set_data(episodes, rewards)
            
            ax2 = self.axes[0, 1]
            ax2.relim()
            ax2.autoscale_view()
        
        # Update Q-Values
        if logger.rl_q_values:
            ax3 = self.axes[1, 0]
            colors_hw = [COLORS['primary'], COLORS['accent'], COLORS['success']]
            
            for hw_id, q_values in logger.rl_q_values.items():
                if len(q_values) > 0:
                    episodes_q = np.arange(len(q_values))
                    if len(q_values) > 20:
                        window = max(1, len(q_values) // 20)
                        smoothed = pd.Series(q_values).rolling(window=window, center=True).mean()
                        y_data = smoothed.values
                    else:
                        y_data = q_values
                    
                    if hw_id not in self.lines['q_values']:
                        color = colors_hw[hw_id % len(colors_hw)]
                        line, = ax3.plot(episodes_q, y_data, '-', linewidth=2, color=color,
                                        label=f'HW {hw_id}', alpha=0.9, markersize=3)
                        self.lines['q_values'][hw_id] = line
                        ax3.legend(loc='upper left', fontsize=10)
                    else:
                        self.lines['q_values'][hw_id].set_data(episodes_q, y_data)
            
            ax3.relim()
            ax3.autoscale_view()
        
        # Update Performance
        if len(logger.episode_latencies) > 0:
            episodes = np.arange(len(logger.episode_latencies))
            latency = np.array(logger.episode_latencies)
            throughput = np.array(logger.episode_throughputs)
            
            self.lines['latency'].set_data(episodes, latency)
            self.lines['throughput'].set_data(episodes, throughput)
            
            ax4 = self.axes[1, 1]
            ax4.relim()
            ax4.autoscale_view()
            # Update twin axis for throughput
            if 'ax4_twin' in self.lines:
                ax4_twin = self.lines['ax4_twin']
                ax4_twin.relim()
                ax4_twin.autoscale_view()
        
        # Update title with current episode
        self.fig.suptitle(f"Dandelion-Learn Training Progress - Episode {episode}", 
                         fontsize=16, fontweight='bold', y=0.995)
        
        # Ensure consistent tick label sizes after updates
        for ax in self.axes.flat:
            ax.tick_params(axis='both', labelsize=PLOT_STYLE['xtick.labelsize'],
                          width=PLOT_STYLE['xtick.major.width'])
        
        # Refresh display
        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)  # Very brief pause for smooth animation
        except Exception as e:
            # If interactive plotting fails, continue silently
            # Plots will still be saved at the end
            pass
    
    def save_final_plots(self, logger: RealTrainingLogger):
        """Save individual plots to files."""
        # Save GNN Loss
        if len(logger.gnn_epochs) > 0:
            fig, ax = plt.subplots(figsize=(12, 7))
            epochs = np.array(logger.gnn_epochs)
            loss = np.array(logger.gnn_losses)
            ax.plot(epochs, loss, 'o-', linewidth=3, color=COLORS['primary'], 
                   markersize=8, label='Training Loss', alpha=0.9, markeredgewidth=1.5,
                   markeredgecolor='white', zorder=3)
            if len(loss) > 5:
                smoothed = savgol_filter(loss, min(5, len(loss)//2*2+1), 3)
                ax.plot(epochs, smoothed, '--', linewidth=2.5, color=COLORS['secondary'], 
                       label='Smoothed', alpha=0.8, zorder=2)
            self._apply_consistent_style(ax, "Training Epoch", "Loss (log scale)", 
                                       "GNN Training Loss Convergence", yscale='log')
            ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, 
                     fontsize=PLOT_STYLE['legend.fontsize'], ncol=1, handlelength=2.5)
            ax.set_axisbelow(True)
            plt.tight_layout()
            plt.savefig(self.results_dir / "gnn_training_loss.png", dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
        
        # Save RL Reward
        if len(logger.rl_episodes) > 0:
            fig, ax = plt.subplots(figsize=(12, 7))
            episodes = np.array(logger.rl_episodes)
            rewards = np.array(logger.rl_rewards)
            ax.plot(episodes, rewards, '-', linewidth=1.5, color=COLORS['secondary'], 
                   alpha=0.4, label='Raw Rewards', zorder=1)
            if len(rewards) > 20:
                window = max(1, len(rewards) // 20)
                smoothed = pd.Series(rewards).rolling(window=window, center=True).mean()
                ax.plot(episodes, smoothed, '-', linewidth=3, color=COLORS['primary'],
                       label='Smoothed', alpha=0.9, zorder=3)
            self._apply_consistent_style(ax, "Training Episode", "Reward (negative latency)", 
                                       "RL Reward Convergence")
            ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, 
                     fontsize=PLOT_STYLE['legend.fontsize'], ncol=1, handlelength=2.5)
            ax.set_axisbelow(True)
            plt.tight_layout()
            plt.savefig(self.results_dir / "rl_reward_convergence.png", dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
        
        # Save Q-Values
        if logger.rl_q_values:
            fig, ax = plt.subplots(figsize=(12, 7))
            colors_hw = [COLORS['primary'], COLORS['accent'], COLORS['success'], '#e377c2', '#7f7f7f']
            markers_hw = ['o', 's', '^', 'D', 'v']
            for hw_id, q_values in logger.rl_q_values.items():
                if len(q_values) > 0:
                    episodes = np.arange(len(q_values))
                    if len(q_values) > 20:
                        window = max(1, len(q_values) // 20)
                        smoothed = pd.Series(q_values).rolling(window=window, center=True).mean()
                    else:
                        smoothed = pd.Series(q_values)
                    color = colors_hw[hw_id % len(colors_hw)]
                    marker = markers_hw[hw_id % len(markers_hw)]
                    ax.plot(episodes, smoothed, '-', linewidth=3, color=color,
                           label=f'Hardware Unit {hw_id}', alpha=0.9, marker=marker,
                           markersize=8, markevery=max(1, len(episodes)//15),
                           markeredgewidth=1.5, markeredgecolor='white', zorder=3)
            self._apply_consistent_style(ax, "Training Episode", "Q-Value", 
                                       "Q-Value Evolution")
            ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, 
                     fontsize=PLOT_STYLE['legend.fontsize'], ncol=1, handlelength=2.5)
            ax.set_axisbelow(True)
            plt.tight_layout()
            plt.savefig(self.results_dir / "q_values_evolution.png", dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
        
        # Save Performance
        if len(logger.episode_latencies) > 0:
            fig, ax = plt.subplots(figsize=(12, 7))
            episodes = np.arange(len(logger.episode_latencies))
            latency = np.array(logger.episode_latencies)
            throughput = np.array(logger.episode_throughputs)
            ax_twin = ax.twinx()
            line1 = ax.plot(episodes, latency, '-', linewidth=3, color=COLORS['primary'],
                           label='Tail Latency (p99)', alpha=0.9, zorder=3)
            line2 = ax_twin.plot(episodes, throughput, '-', linewidth=3, color=COLORS['accent'],
                                label='Throughput', alpha=0.9, zorder=3)
            self._apply_consistent_style(ax, "Training Episode", "Tail Latency (p99, ms)", 
                                     "Performance During Training", twin_ax=ax_twin)
            ax.set_ylabel("Tail Latency (p99, ms)", fontweight='bold', 
                         fontsize=PLOT_STYLE['axes.labelsize'], color=COLORS['primary'])
            ax_twin.set_ylabel("Throughput (jobs/s)", fontweight='bold', 
                             fontsize=PLOT_STYLE['axes.labelsize'], color=COLORS['accent'])
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left', frameon=True, fancybox=True, shadow=True, 
                     fontsize=PLOT_STYLE['legend.fontsize'], ncol=1, handlelength=2.5)
            ax.tick_params(axis='both', labelsize=PLOT_STYLE['xtick.labelsize'],
                          width=PLOT_STYLE['xtick.major.width'])
            ax_twin.tick_params(axis='y', labelsize=PLOT_STYLE['ytick.labelsize'],
                               width=PLOT_STYLE['ytick.major.width'])
            ax.set_axisbelow(True)
            plt.tight_layout()
            plt.savefig(self.results_dir / "performance_during_training.png", dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()


def real_training_simulation(
    root: Path,
    results_dir: Path,
    num_training_episodes: int = 2000,
    jobs_per_episode: int = 100,
) -> RealTrainingLogger:
    """
    Run real training simulation using actual Azure trace data.
    Generates plots in real-time and saves them to results folder.
    """
    logger = RealTrainingLogger()
    
    # Load real Azure trace data
    trace_dir = root / "azure_trace"
    all_jobs: List[InvocationJob] = []
    for day in [1, 2, 3]:
        jobs = load_azure_invocations(trace_dir, day=day, max_jobs=2000)
        all_jobs.extend(jobs)
    
    # Split into training and evaluation sets
    np.random.seed(42)
    np.random.shuffle(all_jobs)
    train_size = int(len(all_jobs) * 0.7)
    train_jobs = all_jobs[:train_size]
    eval_jobs = all_jobs[train_size:]
    
    # Initialize components
    hw_units = build_default_cluster()
    predictor = GNNPredictor()
    scheduler = RLScheduler(hw_units, epsilon=0.1)
    optimizer = MultiObjectiveOptimizer(alpha=1.0, beta=0.05, gamma=0.05)
    
    # Initial GNN training on historical data
    print("\nInitial GNN training on historical data...")
    predictor.fit(train_jobs)
    
    # Compute initial prediction error
    runtime_err, memory_err = compute_prediction_error(predictor, train_jobs)
    initial_loss = (runtime_err + memory_err) / 2.0
    logger.log_gnn_training(0, initial_loss, runtime_err, memory_err)
    print(f"  Initial loss: {initial_loss:.4f}, Runtime error: {runtime_err:.4f}, Memory error: {memory_err:.4f}")
    
    # Online learning episodes
    print(f"\nRunning {num_training_episodes} training episodes...")
    
    # Initialize real-time plotter before the loop
    plotter = None
    try:
        plotter = RealTimePlotter(results_dir)
        plotter.initialize()
        print("\n[OK] Real-time training plot initialized. Training in progress...")
        print("[INFO] If the plot window doesn't appear, plots will be saved at the end.\n")
    except Exception as e:
        print(f"\n[WARNING] Could not initialize real-time plotting: {e}")
        print("[INFO] Training will continue and plots will be saved at the end.\n")
        plotter = None
    
    for episode in range(num_training_episodes):
        # Sample jobs for this episode
        episode_jobs = np.random.choice(
            eval_jobs, 
            size=min(jobs_per_episode, len(eval_jobs)), 
            replace=False
        ).tolist()
        
        # Run simulation with current scheduler
        sim = ClusterSimulator(
            jobs=episode_jobs.copy(),
            hw_units=hw_units,
            predictor=predictor,
            optimizer=optimizer,
            scheduler=scheduler,
            use_optimizer=True,
        )
        
        import time
        start_time = time.time()
        sim.run()
        sim_time = time.time() - start_time
        
        # Initialize default values
        p99_latency = 0.0
        throughput = 0.0
        gpu_util = 0.0
        avg_latency = 0.0
        
        # Compute metrics
        if sim.job_latencies:
            latencies_ms = np.array(sim.job_latencies) * 1000.0
            avg_latency = np.mean(latencies_ms)
            p99_latency = np.percentile(latencies_ms, 99)
            throughput = len(episode_jobs) / sim_time if sim_time > 0 else 0.0
            
            # GPU utilization
            gpu_units = [hw for hw in hw_units if hw.hw_type == "gpu"]
            total_gpu_time = sum(sim.hw_utilization.get(hw.hw_id, 0) for hw in gpu_units)
            gpu_util = total_gpu_time / sim_time if sim_time > 0 else 0.0
            
            logger.log_episode_performance(p99_latency, throughput, gpu_util)
            
            # Log RL updates
            for hw_id in range(len(hw_units)):
                q_value = float(scheduler.values[hw_id])
                reward = -avg_latency
                logger.log_rl_update(episode, hw_id, q_value, reward, p99_latency)
        
        # Periodic GNN fine-tuning (every 20 episodes)
        if episode > 0 and episode % 20 == 0:
            recent_jobs = episode_jobs
            predictor.fit(recent_jobs)
            
            runtime_err, memory_err = compute_prediction_error(predictor, recent_jobs)
            loss = (runtime_err + memory_err) / 2.0
            logger.log_gnn_training(episode // 20, loss, runtime_err, memory_err)
        
        # Update real-time plot every episode for smooth animation
        if plotter is not None:
            try:
                if episode % 1 == 0 or episode == num_training_episodes - 1:
                    plotter.update(logger, episode)
            except Exception as e:
                # If update fails, continue training silently
                pass
        
        # Print progress only every 100 episodes
        if episode > 0 and episode % 100 == 0:
            print(f"  Episode {episode}/{num_training_episodes}: p99={p99_latency:.2f}ms, throughput={throughput:.0f} jobs/s, GPU util={gpu_util:.2%}")
    
    print("\n[OK] Training simulation completed!")
    
    # Save final individual plots
    if plotter is not None:
        try:
            plotter.save_final_plots(logger)
            print("[OK] Saved final individual plots to results folder")
        except Exception as e:
            print(f"[WARNING] Could not save plots: {e}")
    else:
        # If plotter wasn't initialized, create plots from logger data
        print("[INFO] Creating plots from training data...")
        try:
            temp_plotter = RealTimePlotter(results_dir)
            temp_plotter.save_final_plots(logger)
            print("[OK] Saved final individual plots to results folder")
        except Exception as e:
            print(f"[WARNING] Could not create plots: {e}")
    
    return logger


def main():
    """Run real training simulation and save logs."""
    root = Path(".").resolve()
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("TRAINING SIMULATION")
    print("=" * 70)
    
    logger = real_training_simulation(
        root, 
        results_dir,
        num_training_episodes=2000, 
        jobs_per_episode=100
    )
    
    # Save training logs to results folder
    gnn_df = pd.DataFrame({
        "epoch": logger.gnn_epochs,
        "loss": logger.gnn_losses,
        "runtime_error": logger.gnn_runtime_errors,
        "memory_error": logger.gnn_memory_errors,
    })
    gnn_df.to_csv(results_dir / "gnn_training.csv", index=False)
    # Save GNN training logs
    
    rl_df = pd.DataFrame({
        "episode": logger.rl_episodes,
        "reward": logger.rl_rewards,
        "latency": logger.rl_latencies,
    })
    rl_df.to_csv(results_dir / "rl_training.csv", index=False)
    # Save RL training logs
    
    q_values_df = pd.DataFrame(logger.rl_q_values)
    q_values_df.to_csv(results_dir / "rl_q_values.csv", index=False)
    # Save RL Q-values
    
    perf_df = pd.DataFrame({
        "episode": range(len(logger.episode_latencies)),
        "p99_latency": logger.episode_latencies,
        "throughput": logger.episode_throughputs,
        "gpu_utilization": logger.episode_gpu_utils,
    })
    perf_df.to_csv(results_dir / "performance_metrics.csv", index=False)
    # Save performance metrics
    
    print("\n[OK] All training logs and plots saved to results/ folder!")


if __name__ == "__main__":
    main()
