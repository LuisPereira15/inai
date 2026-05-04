"""
evaluate_best_agent.py v5

ALTERAÇÕES v5:
  - Avaliação rápida: se win=0 na dificuldade 0, salta dificuldades superiores
  - Poupa ~75% do tempo quando o agente ainda não completa o nível
  - MAX_STEPS mantido em 2000
  - Comportamento normal quando há wins (avalia todas as dificuldades)

Usage:
    python evaluate_best_agent.py <path_to_best_pkl> [--show] [--full]
    --full: força avaliação em todas as dificuldades mesmo sem wins
"""

import sys
import csv
import pickle as pkl
from pathlib import Path

import numpy as np
import marioai

from agents import MLPAgent
from tasks import MoveForwardTask, HunterTask

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
TASK_CLASS       = MoveForwardTask
DIFFICULTIES     = [0, 3, 5, 10]
N_RUNS_PER_LEVEL = 30
PORT             = 4243
MAX_STEPS        = 2000


# ---------------------------------------------------------------------------
# Run de um episódio
# ---------------------------------------------------------------------------
def run_one_episode(agent, task, exp, difficulty, level_seed):
    task.level_difficulty  = difficulty
    task.env.level_type    = 0
    task.env.level_seed    = level_seed
    rewards  = exp.doEpisodes(1)
    fitness  = sum(rewards[0])
    metrics  = task.get_metrics()
    metrics["fitness"]    = fitness
    metrics["difficulty"] = difficulty
    metrics["level_seed"] = level_seed
    metrics["timeout"]    = not (metrics["won"] or metrics["died"])
    return metrics


# ---------------------------------------------------------------------------
# Avaliação principal
# ---------------------------------------------------------------------------
def evaluate_mlp_agent(pkl_path, show=False, full=False):
    print(f"Loading best agent from: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        best_params = pkl.load(f)

    agent = MLPAgent()
    agent.set_param_vector(best_params)

    task = TASK_CLASS(visualization=show, port=PORT, init_mario_mode=0)
    exp  = marioai.Experiment(task, agent)
    exp.max_fps   = 60 if show else -1
    exp.max_steps = MAX_STEPS

    all_results    = []
    skipped_diffs  = []

    for difficulty in DIFFICULTIES:
        # AVALIAÇÃO RÁPIDA: se dificuldade anterior teve 0 wins, salta
        if not full and skipped_diffs:
            print(f"\n=== Dificuldade {difficulty} — SALTADA "
                  f"(dificuldade {skipped_diffs[-1]} teve 0 wins) ===")
            continue

        print(f"\n=== Difficulty {difficulty} ===")
        diff_results = []

        for run_idx in range(N_RUNS_PER_LEVEL):
            seed    = 1000 * difficulty + run_idx
            metrics = run_one_episode(agent, task, exp, difficulty, seed)
            all_results.append(metrics)
            diff_results.append(metrics)

            outcome = ("WIN" if metrics["won"] else
                      "DEATH" if metrics["died"] else "TIMEOUT")
            print(f"  Run {run_idx+1:>3}/{N_RUNS_PER_LEVEL} "
                  f"(seed={seed}): "
                  f"fitness={metrics['fitness']:>8.2f} | "
                  f"distance={metrics['distance']:>6.1f} | "
                  f"kills={metrics['kills']:>2} | "
                  f"{outcome}")

        # Verificar wins nesta dificuldade
        wins_this_diff = sum(1 for r in diff_results if r["won"])
        print(f"  → Wins nesta dificuldade: {wins_this_diff}/{N_RUNS_PER_LEVEL}")

        # Se 0 wins e não é modo full, marcar para saltar próximas
        if wins_this_diff == 0 and not full:
            skipped_diffs.append(difficulty)
            print(f"  → 0 wins — dificuldades superiores serão saltadas.")

    # Guardar resultados
    Path("data/results").mkdir(parents=True, exist_ok=True)
    pkl_name     = Path(pkl_path).stem
    csv_path     = f"data/results/eval_{pkl_name}.csv"
    summary_path = f"data/results/summary_{pkl_name}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "difficulty", "level_seed", "fitness",
            "distance", "kills", "won", "died",
            "timeout", "final_status", "steps"
        ])
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print_summary_table(all_results)
    save_summary_csv(all_results, summary_path)
    print(f"\nResultados: {csv_path}")
    print(f"Sumário:    {summary_path}")
    if skipped_diffs:
        print(f"Dificuldades saltadas: {skipped_diffs} "
              f"(usa --full para avaliar todas)")


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------
def aggregate_difficulty(results, difficulty):
    subset = [r for r in results if r["difficulty"] == difficulty]
    if not subset:
        return None
    fitnesses = np.array([r["fitness"]  for r in subset])
    distances = np.array([r["distance"] for r in subset])
    kills     = np.array([r["kills"]    for r in subset])
    wins      = sum(1 for r in subset if r["won"])
    deaths    = sum(1 for r in subset if r["died"])
    timeouts  = sum(1 for r in subset if r["timeout"])
    n         = len(subset)
    return {
        "difficulty":     difficulty,
        "runs":           n,
        "win_rate":       wins / n,
        "death_rate":     deaths / n,
        "timeout_rate":   timeouts / n,
        "mean_fitness":   float(fitnesses.mean()),
        "std_fitness":    float(fitnesses.std()),
        "mean_distance":  float(distances.mean()),
        "mean_kills":     float(kills.mean()),
        "total_kills":    int(kills.sum()),
        "total_wins":     wins,
        "total_deaths":   deaths,
        "total_timeouts": timeouts,
    }

def print_summary_table(results):
    evaluated = [d for d in DIFFICULTIES
                 if any(r["difficulty"] == d for r in results)]
    print("\n" + "=" * 105)
    print("FINAL SUMMARY")
    print("=" * 105)
    print(f"{'Diff':>5} | {'Runs':>5} | {'Win%':>6} | {'Death%':>7} | "
          f"{'Timeout%':>9} | {'MeanFit':>10} | {'StdFit':>9} | "
          f"{'MeanDist':>9} | {'MeanKills':>9}")
    print("-" * 105)
    for difficulty in evaluated:
        s = aggregate_difficulty(results, difficulty)
        if s is None: continue
        print(f"{s['difficulty']:>5} | {s['runs']:>5} | "
              f"{100*s['win_rate']:>5.1f}% | {100*s['death_rate']:>6.1f}% | "
              f"{100*s['timeout_rate']:>8.1f}% | {s['mean_fitness']:>10.2f} | "
              f"{s['std_fitness']:>9.2f} | {s['mean_distance']:>9.2f} | "
              f"{s['mean_kills']:>9.2f}")
    print("=" * 105)

def save_summary_csv(results, path):
    fieldnames = ["difficulty", "runs", "win_rate", "death_rate",
                  "timeout_rate", "mean_fitness", "std_fitness",
                  "mean_distance", "mean_kills",
                  "total_kills", "total_wins", "total_deaths", "total_timeouts"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for difficulty in DIFFICULTIES:
            s = aggregate_difficulty(results, difficulty)
            if s is not None:
                writer.writerow(s)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_best_agent.py <pkl> [--show] [--full]")
        sys.exit(1)

    pkl_path = sys.argv[1]
    show     = "--show" in sys.argv
    full     = "--full" in sys.argv
    evaluate_mlp_agent(pkl_path, show=show, full=full)
