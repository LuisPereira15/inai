"""
hyperparameter_search.py
========================
Testa combinações de hiperparâmetros do EA de forma sistemática.
Cada configuração corre 1 seed por 30 gerações (rápido) e guardamos
o best fitness final para comparar.

Diagnóstico dos resultados anteriores:
- Seeds 1-5 atingem 90% do best entre gerações 57-97 → ainda há margem
- Seed 11 colapsou completamente (std=0 desde geração ~20) → diversidade a zero
- Win rate = 0% em todas as seeds → agente não termina o nível
- Distância média ~300-550px de ~3000px totais → só chega a 15-18% do nível

Problemas identificados:
1. Convergência prematura (seed 11 e std baixo nas seeds 4)
2. Agente não aprende a saltar obstáculos / terminar o nível
3. Possível problema na fitness: recompensa de progresso não é suficiente

O que este script testa:
- MU/LAMBDA: tamanho da população (mais população = mais diversidade)
- SIGMA_MUT: intensidade da mutação
- TOURNAMENT_K: pressão de seleção
- WIN_BONUS: peso do bónus de win na fitness

Uso:
    python hyperparameter_search.py

Resultados guardados em: data/hypersearch/hypersearch_results.csv
"""

import sys
import csv
import time
import pickle as pkl
import itertools
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from agents.mlp_agent import MLPAgent
from evaluation import evaluate_population

# ---------------------------------------------------------------------------
# Configurações a testar
# Mantemos poucas combinações para ser rápido (~30 gens cada)
# ---------------------------------------------------------------------------
SEARCH_SPACE = {
    "MU":           [30, 50, 80],        # tamanho população
    "LAMBDA":       [30, 50],            # offspring por geração  
    "SIGMA_MUT":    [0.05, 0.1, 0.2],   # intensidade mutação
    "TOURNAMENT_K": [3, 5],             # pressão seleção
}

EVAL_GENERATIONS = 30   # gerações por configuração (rápido)
EVAL_SEED = 42          # seed fixa para comparação justa
CROSSOVER_PROB = 0.8
MUTATION_PROB = 1.0
SIGMA_INIT = 0.5
ADAPTIVE_GEN = 15       # metade de EVAL_GENERATIONS


# ---------------------------------------------------------------------------
# Operadores evolutivos (iguais ao evolution.py)
# ---------------------------------------------------------------------------
def init_population(pop_size, num_params, sigma_init):
    return [np.random.randn(num_params).astype(np.float32) * sigma_init
            for _ in range(pop_size)]


def tournament_selection(population, fitnesses, k):
    indices = np.random.randint(0, len(population), size=k)
    best = indices[0]
    for idx in indices[1:]:
        if fitnesses[idx] > fitnesses[best]:
            best = idx
    return deepcopy(population[best])


def blx_alpha_crossover(p1, p2, alpha=0.5):
    lo = np.minimum(p1, p2)
    hi = np.maximum(p1, p2)
    diff = hi - lo
    child = np.random.uniform(lo - alpha * diff, hi + alpha * diff).astype(np.float32)
    return child


def gaussian_mutation(ind, sigma):
    return ind + np.random.randn(len(ind)).astype(np.float32) * sigma


def make_offspring(parents, fitnesses, n, sigma, k):
    offspring = []
    for _ in range(n):
        p1 = tournament_selection(parents, fitnesses, k)
        p2 = tournament_selection(parents, fitnesses, k)
        child = blx_alpha_crossover(p1, p2) if np.random.rand() < CROSSOVER_PROB else deepcopy(p1)
        if np.random.rand() < MUTATION_PROB:
            child = gaussian_mutation(child, sigma)
        offspring.append(child)
    return offspring


def survivor_selection(parents, pf, offspring, of, mu):
    combined = parents + offspring
    fits = np.concatenate([pf, of])
    best = np.argsort(-fits)[:mu]
    return [combined[i] for i in best], fits[best]


# ---------------------------------------------------------------------------
# Correr uma configuração
# ---------------------------------------------------------------------------
def run_config(config, num_params):
    MU = config["MU"]
    LAMBDA = config["LAMBDA"]
    SIGMA = config["SIGMA_MUT"]
    K = config["TOURNAMENT_K"]

    population = init_population(MU, num_params, SIGMA_INIT)
    fitnesses = evaluate_population(MLPAgent, population)

    best_of_run = float(fitnesses.max())
    history = []

    for gen in range(EVAL_GENERATIONS):
        # Sigma adaptativo
        sigma = SIGMA / 2 if gen >= ADAPTIVE_GEN else SIGMA

        offspring = make_offspring(population, fitnesses, LAMBDA, sigma, K)
        offspring_fits = evaluate_population(MLPAgent, offspring)

        population, fitnesses = survivor_selection(
            population, fitnesses, offspring, offspring_fits, MU
        )

        gen_best = float(fitnesses.max())
        gen_mean = float(fitnesses.mean())
        gen_std = float(fitnesses.std())

        if gen_best > best_of_run:
            best_of_run = gen_best

        history.append({
            "gen": gen + 1,
            "best": gen_best,
            "mean": gen_mean,
            "std": gen_std,
            "best_of_run": best_of_run,
        })

        print(f"  Gen {gen+1:3d}/{EVAL_GENERATIONS} | "
              f"best={gen_best:.1f} | mean={gen_mean:.1f} | "
              f"std={gen_std:.1f} | best_run={best_of_run:.1f}")

    return best_of_run, history


# ---------------------------------------------------------------------------
# Main: grid search
# ---------------------------------------------------------------------------
def main():
    np.random.seed(EVAL_SEED)
    torch.manual_seed(EVAL_SEED)

    template = MLPAgent()
    num_params = len(template.get_param_vector())
    print(f"Número de parâmetros MLP: {num_params}")

    Path("data/hypersearch").mkdir(parents=True, exist_ok=True)

    # Gerar todas as combinações
    keys = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    all_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"\nTotal de configurações a testar: {len(all_configs)}")
    print(f"Gerações por configuração: {EVAL_GENERATIONS}")
    print(f"Tempo estimado: ~{len(all_configs) * 3:.0f} minutos\n")

    results = []
    csv_path = "data/hypersearch/hypersearch_results.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config_id", "MU", "LAMBDA", "SIGMA_MUT",
                         "TOURNAMENT_K", "best_fitness", "time_seconds"])

        for i, config in enumerate(all_configs):
            print(f"\n{'='*60}")
            print(f"Config {i+1}/{len(all_configs)}: {config}")
            print(f"{'='*60}")

            # Reset seed para cada config ser comparável
            np.random.seed(EVAL_SEED)
            torch.manual_seed(EVAL_SEED)

            t0 = time.perf_counter()
            best, history = run_config(config, num_params)
            elapsed = time.perf_counter() - t0

            results.append({**config, "best_fitness": best, "time": elapsed})
            writer.writerow([i + 1, config["MU"], config["LAMBDA"],
                             config["SIGMA_MUT"], config["TOURNAMENT_K"],
                             best, elapsed])
            f.flush()

            # Guardar histórico desta config
            hist_path = f"data/hypersearch/history_config_{i+1}.csv"
            with open(hist_path, "w", newline="") as hf:
                hw = csv.DictWriter(hf, fieldnames=["gen", "best", "mean", "std", "best_of_run"])
                hw.writeheader()
                hw.writerows(history)

            print(f"  → Best fitness: {best:.2f} em {elapsed:.0f}s")

    # Ranking final
    results.sort(key=lambda x: x["best_fitness"], reverse=True)
    print(f"\n{'='*60}")
    print("TOP 5 CONFIGURAÇÕES:")
    print(f"{'='*60}")
    for rank, r in enumerate(results[:5], 1):
        print(f"#{rank}: MU={r['MU']} LAMBDA={r['LAMBDA']} "
              f"SIGMA={r['SIGMA_MUT']} K={r['TOURNAMENT_K']} "
              f"→ best={r['best_fitness']:.2f}")

    print(f"\nResultados completos guardados em: {csv_path}")

    # Guardar melhor config
    best_config = results[0]
    print(f"\nMELHOR CONFIGURAÇÃO ENCONTRADA:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")

    return best_config


if __name__ == "__main__":
    main()
