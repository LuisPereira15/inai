"""
evaluate_best_agent.py — Avaliação final do melhor agente evoluído.

Para cada nível testado imprime no terminal:
    WIN | TIMEOUT | DEATH

e guarda os resultados em CSV para análise posterior.

Uso:
    python evaluate_best_agent.py <path_to_pkl>           # sem visualização
    python evaluate_best_agent.py <path_to_pkl> --show    # com janela gráfica

Exemplos:
    python evaluate_best_agent.py data/mlp_best_agents/es_dev_seed_1_850.000.pkl
    python evaluate_best_agent.py data/mlp_best_agents/es_dev_seed_1_850.000.pkl --show

Configuração rápida para testes DEV:
    Alterar N_RUNS_PER_LEVEL para 5 e DIFFICULTIES para [0]
    para uma avaliação rápida de ~2-3 min.
"""

import sys
import csv
import pickle as pkl
from pathlib import Path

import numpy as np
import torch
import marioai

from agents import MLPAgent
from tasks import MoveForwardTask, HunterTask


# ---------------------------------------------------------------------------
# Configuração — ajusta aqui para DEV (rápido) ou FULL (entrega)
# ---------------------------------------------------------------------------
TASK_CLASS = MoveForwardTask

# Para testes iniciais (DEV): só difficulty=0, 10 runs
# Para entrega (FULL): [0, 3, 5, 10], 30 runs
DIFFICULTIES      = [0]          # DEV: só o fácil para ver WIN/TIMEOUT rápido
N_RUNS_PER_LEVEL  = 10           # DEV: 10 runs (~2-3 min)
# DIFFICULTIES     = [0, 3, 5, 10]  # FULL: descomentar para entrega
# N_RUNS_PER_LEVEL = 30             # FULL: descomentar para entrega

PORT = 4245   # porta docker reservada para avaliação (não conflitua com treino)


# ---------------------------------------------------------------------------
# Correr um episódio e devolver métricas
# ---------------------------------------------------------------------------
def run_one_episode(agent, task, exp, difficulty, level_seed):
    task.level_difficulty = difficulty
    task.env.level_type   = 0
    task.env.level_seed   = level_seed

    rewards = exp.doEpisodes(1)
    fitness = sum(rewards[0])

    metrics = task.get_metrics()
    metrics["fitness"]    = fitness
    metrics["difficulty"] = difficulty
    metrics["level_seed"] = level_seed
    metrics["timeout"]    = not (metrics["won"] or metrics["died"])
    return metrics


# ---------------------------------------------------------------------------
# Avaliação principal
# ---------------------------------------------------------------------------
def evaluate_mlp_agent(pkl_path, show=False):
    print(f"\nLoading: {pkl_path}")
    with open(pkl_path, "rb") as f:
        best_params = pkl.load(f)

    agent = MLPAgent()
    agent.set_param_vector(best_params)
    agent.stochastic = False   # eval determinístico — comportamento reproduzível

    task = TASK_CLASS(visualization=show, port=PORT, init_mario_mode=0)
    exp  = marioai.Experiment(task, agent)
    exp.max_fps = 60 if show else -1

    all_results = []

    for difficulty in DIFFICULTIES:
        print(f"\n{'='*55}")
        print(f"  Difficulty {difficulty}  ({N_RUNS_PER_LEVEL} runs)")
        print(f"{'='*55}")

        for run_idx in range(N_RUNS_PER_LEVEL):
            seed    = 1000 * difficulty + run_idx
            metrics = run_one_episode(agent, task, exp, difficulty, seed)
            all_results.append(metrics)

            # Resultado claro no terminal
            if metrics["won"]:
                outcome = "WIN    ✓"
            elif metrics["died"]:
                outcome = "DEATH  ✗"
            else:
                outcome = "TIMEOUT ·"

            print(
                f"  Run {run_idx+1:>2}/{N_RUNS_PER_LEVEL} "
                f"(seed={seed:>5}) | "
                f"fitness={metrics['fitness']:>8.1f} | "
                f"dist={metrics['distance']:>6.1f} | "
                f"kills={metrics['kills']:>2} | "
                f"{outcome}"
            )

    # Guardar CSV
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

    # Sumário no terminal
    print_summary(all_results)
    print(f"\nCSV guardado: {csv_path}")


# ---------------------------------------------------------------------------
# Sumário
# ---------------------------------------------------------------------------
def print_summary(results):
    print(f"\n{'='*55}")
    print("  SUMÁRIO")
    print(f"{'='*55}")
    print(f"  {'Diff':>4} | {'Runs':>4} | "
          f"{'Win%':>5} | {'Death%':>6} | {'Timeout%':>8} | "
          f"{'AvgDist':>7} | {'AvgFit':>8}")
    print(f"  {'-'*53}")

    for diff in sorted(set(r["difficulty"] for r in results)):
        sub = [r for r in results if r["difficulty"] == diff]
        n   = len(sub)
        wins    = sum(1 for r in sub if r["won"])
        deaths  = sum(1 for r in sub if r["died"])
        timeouts = sum(1 for r in sub if r["timeout"])
        avg_dist = np.mean([r["distance"] for r in sub])
        avg_fit  = np.mean([r["fitness"]  for r in sub])
        print(f"  {diff:>4} | {n:>4} | "
              f"{100*wins/n:>4.0f}% | {100*deaths/n:>5.0f}% | "
              f"{100*timeouts/n:>7.0f}% | "
              f"{avg_dist:>7.1f} | {avg_fit:>8.1f}")

    # Totais
    n     = len(results)
    wins  = sum(1 for r in results if r["won"])
    print(f"  {'-'*53}")
    print(f"  TOTAL: {wins}/{n} WINs ({100*wins/n:.1f}%)")
    print(f"{'='*55}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage: python evaluate_best_agent.py <pkl_path> [--show]")
        sys.exit(1)

    pkl_path = args[0]
    show     = "--show" in args

    evaluate_mlp_agent(pkl_path, show=show)
