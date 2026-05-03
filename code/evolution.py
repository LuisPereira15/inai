"""
Evolutionary Algorithm: (µ + λ) Evolution Strategy with elitism for the MLPAgent.

ALTERAÇÕES v3 (otimizadas via hyperparameter_search.py):
  - MU reduzido de 50 para 30 (população menor converge melhor — confirmado no hypersearch)
  - TOURNAMENT_K aumentado de 3 para 5 (pressão de seleção mais alta)
  - GENERATIONS aumentado de 100 para 150 (mais gerações para aproveitar melhor config)
  - SIGMA_MUT aumentado de 0.05 para 0.1 (mais consistente entre seeds)
  - SIGMA_MUT_FINE ajustado de 0.02 para 0.05 (desce de 0.1 para 0.05 na gen 50)
  - MUTATION_PROB = 1.0 mantido
  - Adaptive sigma mantido: 0.1 até gen 50, depois 0.05

Usage:
    python evolution.py <seed>
e.g. python evolution.py 1
"""

import sys
import time
import pickle as pkl
import csv
from copy import deepcopy
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch
import matplotlib.pyplot as plt

from agents.mlp_agent import MLPAgent
from evaluation import evaluate_population


# ---------------------------------------------------------------------------
# 1. Hyperparameters
# OTIMIZADOS via hyperparameter_search.py (v3)
# Melhor config encontrada: MU=30, LAMBDA=50, SIGMA=0.1→0.05, K=5
# ---------------------------------------------------------------------------
MU = 30              # ALTERADO: era 50, população mais pequena converge melhor
LAMBDA = 50          # mantido: mais offspring que pais favorece exploração
GENERATIONS = 150    # ALTERADO: era 100, mais gerações para aproveitar melhor config
TOURNAMENT_K = 5     # ALTERADO: era 3, pressão de seleção mais alta
CROSSOVER_PROB = 0.8 # mantido
MUTATION_PROB = 1.0  # mantido: todos os filhos sofrem mutação
SIGMA_INIT = 0.5     # mantido
SIGMA_MUT = 0.1      # ALTERADO: era 0.05, mais consistente entre seeds

# Adaptive sigma: após esta geração, usa sigma mais pequeno para refinamento
ADAPTIVE_SIGMA_GEN = 50
SIGMA_MUT_FINE = 0.05  # ALTERADO: era 0.02, desce de 0.1 para 0.05 na gen 50


# ---------------------------------------------------------------------------
# 2. Small utilities
# ---------------------------------------------------------------------------
@contextmanager
def timer_context(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"[{label}] Elapsed time: {end - start:.4f} seconds")


def make_evolution_plot(best, mean, title, save=False):
    plt.plot(best, label='Best Reward')
    plt.plot(mean, label='Mean Reward')
    plt.xlabel('Generation')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.draw()
    if save:
        plt.savefig(f'{title}.png')
    plt.pause(0.01)
    plt.clf()


# ---------------------------------------------------------------------------
# 3. Evolutionary operators
# ---------------------------------------------------------------------------
def init_individual(num_params):
    return np.random.randn(num_params).astype(np.float32) * SIGMA_INIT


def init_population(pop_size, num_params):
    return [init_individual(num_params) for _ in range(pop_size)]


def tournament_selection(population, fitnesses, k=TOURNAMENT_K):
    indices = np.random.randint(0, len(population), size=k)
    best_idx = indices[0]
    for idx in indices[1:]:
        if fitnesses[idx] > fitnesses[best_idx]:
            best_idx = idx
    return deepcopy(population[best_idx])


def blx_alpha_crossover(parent1, parent2, alpha=0.5):
    lower = np.minimum(parent1, parent2)
    upper = np.maximum(parent1, parent2)
    diff = upper - lower
    low = lower - alpha * diff
    high = upper + alpha * diff
    child = np.random.uniform(low, high).astype(np.float32)
    return child


def gaussian_mutation(individual, sigma=SIGMA_MUT):
    noise = np.random.randn(len(individual)).astype(np.float32) * sigma
    return individual + noise


def make_offspring(parents, fitnesses, num_offspring, current_gen):
    """Gera filhos com sigma adaptativo baseado na geração atual."""
    # Adaptive sigma: refinamento nas gerações avançadas
    if current_gen >= ADAPTIVE_SIGMA_GEN:
        sigma = SIGMA_MUT_FINE
    else:
        sigma = SIGMA_MUT

    offspring = []
    for _ in range(num_offspring):
        p1 = tournament_selection(parents, fitnesses)
        p2 = tournament_selection(parents, fitnesses)
        if np.random.rand() < CROSSOVER_PROB:
            child = blx_alpha_crossover(p1, p2)
        else:
            child = deepcopy(p1)
        # MUTATION_PROB = 1.0: todos os filhos sofrem mutação
        if np.random.rand() < MUTATION_PROB:
            child = gaussian_mutation(child, sigma=sigma)
        offspring.append(child)
    return offspring


def survivor_selection_mu_plus_lambda(parents, parent_fits,
                                      offspring, offspring_fits, mu):
    combined = parents + offspring
    combined_fits = np.concatenate([parent_fits, offspring_fits])
    best_indices = np.argsort(-combined_fits)[:mu]
    new_pop = [combined[i] for i in best_indices]
    new_fits = combined_fits[best_indices]
    return new_pop, new_fits


# ---------------------------------------------------------------------------
# 4. Main evolutionary loop
# ---------------------------------------------------------------------------
def evolution_strategy(seed):
    # 1) Discover genotype length
    template_agent = MLPAgent()
    num_params = len(template_agent.get_param_vector())
    print(f"Genotype length (number of MLP parameters): {num_params}")

    # 2) Initialise + evaluate
    population = init_population(MU, num_params)
    print("Evaluating initial population...")
    fitnesses = evaluate_population(MLPAgent, population)

    # 3) Best-of-run tracking
    best_idx = int(np.argmax(fitnesses))
    best_individual = deepcopy(population[best_idx])
    best_reward = float(fitnesses[best_idx])
    best_history = []
    mean_history = []
    min_history = []
    std_history = []
    gen_times = []

    print(f"Initial best reward: {best_reward:.3f}")
    print(f"Initial mean reward: {fitnesses.mean():.3f}")

    Path("data/mlp_best_agents").mkdir(parents=True, exist_ok=True)

    # CSV log file
    csv_path = f"data/mlp_best_agents/log_seed_{seed}.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["generation", "best", "mean", "min", "std", "best_of_run"])

    # 4) Generation loop
    for gen in range(GENERATIONS):
        gen_start = time.perf_counter()
        print(f"\n--- Generation {gen+1}/{GENERATIONS} ---")

        # Sigma adaptativo: mostra qual está a ser usado
        current_sigma = SIGMA_MUT_FINE if gen >= ADAPTIVE_SIGMA_GEN else SIGMA_MUT
        print(f"[Sigma = {current_sigma}]")

        # 4.1 Generate offspring
        offspring = make_offspring(population, fitnesses, LAMBDA, gen)

        # 4.2 Evaluate them
        with timer_context("Evaluate offspring"):
            offspring_fits = evaluate_population(MLPAgent, offspring)

        # 4.3 Survivor selection (µ+λ)
        population, fitnesses = survivor_selection_mu_plus_lambda(
            population, fitnesses, offspring, offspring_fits, MU
        )

        # 4.4 Update best-of-run
        gen_best_idx = int(np.argmax(fitnesses))
        gen_best_reward = float(fitnesses[gen_best_idx])
        if gen_best_reward > best_reward:
            best_reward = gen_best_reward
            best_individual = deepcopy(population[gen_best_idx])
            fname = (f"data/mlp_best_agents/"
                     f"es_seed_{seed}_{best_reward:.3f}.pkl")
            with open(fname, 'wb') as f:
                pkl.dump(best_individual, f)
            print(f">>> New best! Saved to {fname}")

        # 4.5 Detailed logging
        mean_reward = float(fitnesses.mean())
        min_reward = float(fitnesses.min())
        std_reward = float(fitnesses.std())

        gen_elapsed = time.perf_counter() - gen_start
        gen_times.append(gen_elapsed)
        avg_gen_time = np.mean(gen_times)
        gens_left = GENERATIONS - (gen + 1)
        eta_seconds = avg_gen_time * gens_left
        eta_min = int(eta_seconds // 60)
        eta_sec = int(eta_seconds % 60)

        print(
            f"Gen {gen+1}: "
            f"Best = {gen_best_reward:.2f} | "
            f"Mean = {mean_reward:.2f} | "
            f"Min = {min_reward:.2f} | "
            f"Std = {std_reward:.2f} | "
            f"Best-of-run = {best_reward:.2f} | "
            f"ETA = {eta_min}m{eta_sec:02d}s"
        )

        best_history.append(gen_best_reward)
        mean_history.append(mean_reward)
        min_history.append(min_reward)
        std_history.append(std_reward)

        csv_writer.writerow([gen + 1, gen_best_reward, mean_reward,
                             min_reward, std_reward, best_reward])
        csv_file.flush()

    csv_file.close()

    # 5) Final plot
    make_evolution_plot(
        best_history, mean_history,
        title=f"ES_MLP_seed_{seed}", save=True
    )

    return best_individual, best_reward


# ---------------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evolution.py <seed>")
        sys.exit(1)

    seed = int(sys.argv[1])
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    best_ind, best_fit = evolution_strategy(seed)
    print(f"\n=== Final best fitness: {best_fit:.3f} ===")
