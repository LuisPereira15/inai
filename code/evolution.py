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
# ---------------------------------------------------------------------------
MU = 15              # number of parents (elite survivors)
LAMBDA = 50          # number of offspring generated per generation
GENERATIONS = 100    # number of generations
TOURNAMENT_K = 2     # tournament size for parent selection
CROSSOVER_PROB = 0.7 # probability of applying crossover
MUTATION_PROB = 1.0  # all offspring undergo mutation
SIGMA_INIT = 0.5     # std-dev of the initial random weights

# Mutation Step Decay
SIGMA_STAGE_1 = 0.5   # Gen 0  to 99  (Max Exploration)
SIGMA_STAGE_2 = 0.35  # Gen 100 to 199 (Moderate Exploration)
SIGMA_STAGE_3 = 0.20  # Gen 200 to 349 (Initial Tuning)
SIGMA_STAGE_4 = 0.05  # Gen 350+       (Surgical Refinement)


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


def gaussian_mutation(individual, sigma):
    noise = np.random.randn(len(individual)).astype(np.float32) * sigma
    return individual + noise


def make_offspring(parents, fitnesses, num_offspring, current_sigma):
    offspring = []
    for _ in range(num_offspring):
        p1 = tournament_selection(parents, fitnesses)
        p2 = tournament_selection(parents, fitnesses)

        if np.random.rand() < CROSSOVER_PROB:
            child = blx_alpha_crossover(p1, p2)
        else:
            child = deepcopy(p1)

        if np.random.rand() < MUTATION_PROB:
            child = gaussian_mutation(child, sigma=current_sigma)

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

    # 2) Draw seeds for the initial evaluation
    # Each generation gets 3 random seeds, shared across the whole population
    # so all individuals are evaluated under the same conditions.
    gen_seeds = np.random.randint(1, 1000, size=3).tolist()
    print(f"Initial evaluation seeds: {gen_seeds}")

    # 3) Initialise + evaluate
    population = init_population(MU, num_params)
    print("Evaluating initial population...")
    fitnesses = evaluate_population(MLPAgent, population, seeds=gen_seeds)

    # 4) Best-of-run tracking
    best_idx = int(np.argmax(fitnesses))
    best_individual = deepcopy(population[best_idx])
    best_reward = float(fitnesses[best_idx])
    best_history = []
    mean_history = []
    min_history = []
    std_history = []

    print(f"Initial best reward: {best_reward:.3f}")
    print(f"Initial mean reward: {fitnesses.mean():.3f}")

    Path("data/mlp_best_agents").mkdir(parents=True, exist_ok=True)

    csv_path = f"data/mlp_best_agents/log_seed_{seed}.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["generation", "best", "mean", "min", "std",
                         "best_of_run", "gen_seeds"])

    # 5) Generation loop
    for gen in range(GENERATIONS):
        print(f"\n--- Generation {gen+1}/{GENERATIONS} ---")

        # Draw 3 new random seeds for this generation.
        # All individuals in this generation are evaluated on the same 3 seeds,
        # ensuring fair comparison while varying across generations.
        gen_seeds = np.random.randint(1, 1000, size=3).tolist()
        print(f"Generation seeds: {gen_seeds}")

        # Sigma schedule
        if gen < 100:
            current_sigma = SIGMA_STAGE_1
        elif gen < 200:
            current_sigma = SIGMA_STAGE_2
        elif gen < 350:
            current_sigma = SIGMA_STAGE_3
        else:
            current_sigma = SIGMA_STAGE_4

        print(f"[Sigma = {current_sigma}]")

        # 5.1 Generate offspring
        offspring = make_offspring(population, fitnesses, LAMBDA, current_sigma)

        # 5.2 Evaluate offspring
        with timer_context("Evaluate offspring"):
            offspring_fits = evaluate_population(MLPAgent, offspring, seeds=gen_seeds)

        # 5.3 Survivor selection (µ+λ)
        population, fitnesses = survivor_selection_mu_plus_lambda(
            population, fitnesses, offspring, offspring_fits, MU
        )

        # 5.4 Elitism: guarantee the best-of-run champion enters next generation
        # untouched, regardless of what survivor selection picked.
        population[0] = deepcopy(best_individual)
        fitnesses[0] = best_reward

        # 5.5 Update best-of-run
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

        # 5.6 Logging
        mean_reward = float(fitnesses.mean())
        min_reward = float(fitnesses.min())
        std_reward = float(fitnesses.std())

        print(
            f"Gen {gen+1}: "
            f"Best = {gen_best_reward:.2f} | "
            f"Mean = {mean_reward:.2f} | "
            f"Min = {min_reward:.2f} | "
            f"Std = {std_reward:.2f} | "
            f"Best-of-run = {best_reward:.2f}"
        )

        best_history.append(gen_best_reward)
        mean_history.append(mean_reward)
        min_history.append(min_reward)
        std_history.append(std_reward)

        csv_writer.writerow([gen + 1, gen_best_reward, mean_reward,
                             min_reward, std_reward, best_reward,
                             str(gen_seeds)])
        csv_file.flush()

    csv_file.close()

    # 6) Final plot
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