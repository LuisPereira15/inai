"""
Final assessment of the best evolved MLP agent.

This version reads the OFFICIAL final status of each episode from
task.status (set by the engine when the episode ends), so Win% and
Death% are now reported correctly. Episodes that end without dying
or finishing are counted as "timeout" (Mario got stuck or ran out
of time).

Usage:
    python evaluate_best_agent.py <path_to_best_pkl> [--show]

Examples:
    python evaluate_best_agent.py data/mlp_best_agents/es_seed_1_2740.850.pkl
    python evaluate_best_agent.py data/mlp_best_agents/es_seed_1_2740.850.pkl --show
"""

import sys
import csv
import time
import pickle as pkl
from pathlib import Path

import numpy as np
import torch
import marioai

from agents import MLPAgent
from tasks import MoveForwardTask, HunterTask


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
TASK_CLASS = MoveForwardTask         # change to HunterTask for Stage 2
DIFFICULTIES = [0, 3, 5, 10]         # same as the paper
N_RUNS_PER_LEVEL = 30                # number of seeds per difficulty
PORT = 4245                          # one of the ports docker exposes


# ---------------------------------------------------------------------------
# 2. Helper: run ONE episode and read metrics
# ---------------------------------------------------------------------------
def run_one_episode(agent, task, exp, difficulty, level_seed):
    """Run a single episode and return a dict with the metrics."""
    # Configure the level
    task.level_difficulty = difficulty
    task.env.level_type = 0
    task.env.level_seed = level_seed

    # Run the episode (this calls task.reset() internally, which clears metrics)
    rewards = exp.doEpisodes(1)
    fitness = sum(rewards[0])

    # IMPORTANT: by this point task.status reflects the FINAL status:
    #   1 = Mario reached the flag (WIN)
    #   2 = Mario died (DEATH)
    #   other (usually 0) = ran out of time / got stuck (TIMEOUT)
    metrics = task.get_metrics()
    metrics["fitness"] = fitness
    metrics["difficulty"] = difficulty
    metrics["level_seed"] = level_seed
    # Anything that is neither a win nor a death is a timeout
    metrics["timeout"] = not (metrics["won"] or metrics["died"])
    return metrics


# ---------------------------------------------------------------------------
# 3. Main evaluation routine
# ---------------------------------------------------------------------------
def evaluate_mlp_agent(pkl_path, show=False):
    print(f"Loading best agent from: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        best_params = pkl.load(f)

    # Build the agent and load the evolved weights
    agent = MLPAgent()
    agent.set_param_vector(best_params)

    # Build the task / experiment
    task = TASK_CLASS(visualization=show, port=PORT, init_mario_mode=0)
    exp = marioai.Experiment(task, agent)
    exp.max_fps = 60 if show else -1

    # Loop over difficulties and seeds
    all_results = []
    for difficulty in DIFFICULTIES:
        print(f"\n=== Difficulty {difficulty} ===")
        for run_idx in range(N_RUNS_PER_LEVEL):
            seed = 1000 * difficulty + run_idx
            metrics = run_one_episode(agent, task, exp, difficulty, seed)
            all_results.append(metrics)

            # Friendly outcome label for logging
            if metrics["won"]:
                outcome = "WIN"
            elif metrics["died"]:
                outcome = "DEATH"
            else:
                outcome = "TIMEOUT"

            print(
                f"  Run {run_idx+1:>3}/{N_RUNS_PER_LEVEL} "
                f"(seed={seed}): "
                f"fitness={metrics['fitness']:>8.2f} | "
                f"distance={metrics['distance']:>6.1f} | "
                f"kills={metrics['kills']:>2} | "
                f"status={metrics['final_status']} -> {outcome}"
            )

    # Save raw per-run results
    Path("data/results").mkdir(parents=True, exist_ok=True)
    pkl_name = Path(pkl_path).stem
    csv_path = f"data/results/eval_{pkl_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["difficulty", "level_seed", "fitness",
                        "distance", "kills", "won", "died", "timeout",
                        "final_status", "steps"]
        )
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
    print(f"\nRaw results saved to: {csv_path}")

    # Print and save a clean summary table
    print_summary_table(all_results)
    summary_path = f"data/results/summary_{pkl_name}.csv"
    save_summary_csv(all_results, summary_path)
    print(f"Summary saved to:    {summary_path}")


# ---------------------------------------------------------------------------
# 4. Summary helpers
# ---------------------------------------------------------------------------
def aggregate_difficulty(results, difficulty):
    """Compute the aggregated stats for one difficulty."""
    subset = [r for r in results if r["difficulty"] == difficulty]
    if not subset:
        return None

    fitnesses = np.array([r["fitness"] for r in subset])
    distances = np.array([r["distance"] for r in subset])
    kills = np.array([r["kills"] for r in subset])
    wins = sum(1 for r in subset if r["won"])
    deaths = sum(1 for r in subset if r["died"])
    timeouts = sum(1 for r in subset if r["timeout"])
    n = len(subset)

    return {
        "difficulty": difficulty,
        "runs": n,
        "win_rate": wins / n,
        "death_rate": deaths / n,
        "timeout_rate": timeouts / n,
        "mean_fitness": float(fitnesses.mean()),
        "std_fitness": float(fitnesses.std()),
        "mean_distance": float(distances.mean()),
        "mean_kills": float(kills.mean()),
        "total_kills": int(kills.sum()),
        "total_wins": wins,
        "total_deaths": deaths,
        "total_timeouts": timeouts,
    }


def print_summary_table(results):
    """Pretty-print the summary table to the terminal."""
    print("\n" + "=" * 105)
    print("FINAL SUMMARY (averaged over runs per difficulty)")
    print("=" * 105)
    header = (
        f"{'Diff':>5} | {'Runs':>5} | "
        f"{'Win%':>6} | {'Death%':>7} | {'Timeout%':>9} | "
        f"{'MeanFit':>10} | {'StdFit':>9} | "
        f"{'MeanDist':>9} | {'MeanKills':>9}"
    )
    print(header)
    print("-" * 105)
    for difficulty in DIFFICULTIES:
        s = aggregate_difficulty(results, difficulty)
        if s is None:
            continue
        print(
            f"{s['difficulty']:>5} | "
            f"{s['runs']:>5} | "
            f"{100*s['win_rate']:>5.1f}% | "
            f"{100*s['death_rate']:>6.1f}% | "
            f"{100*s['timeout_rate']:>8.1f}% | "
            f"{s['mean_fitness']:>10.2f} | "
            f"{s['std_fitness']:>9.2f} | "
            f"{s['mean_distance']:>9.2f} | "
            f"{s['mean_kills']:>9.2f}"
        )
    print("=" * 105)


def save_summary_csv(results, path):
    """Save the per-difficulty summary to a CSV file."""
    fieldnames = ["difficulty", "runs", "win_rate", "death_rate",
                  "timeout_rate", "mean_fitness", "std_fitness",
                  "mean_distance", "mean_kills",
                  "total_kills", "total_wins", "total_deaths",
                  "total_timeouts"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for difficulty in DIFFICULTIES:
            s = aggregate_difficulty(results, difficulty)
            if s is not None:
                writer.writerow(s)


# ---------------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_best_agent.py <path_to_best_pkl> [--show]")
        print("Example: python evaluate_best_agent.py "
              "data/mlp_best_agents/es_seed_1_2740.850.pkl")
        sys.exit(1)

    pkl_path = sys.argv[1]
    show = (len(sys.argv) >= 3 and sys.argv[2] == "--show")

    evaluate_mlp_agent(pkl_path, show=show)