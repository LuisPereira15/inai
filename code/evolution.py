"""
evolution.py — μ+λ Evolution Strategy para Super Mario (MLP NeuroEvolution)

Versão 4 — run limpa com todas as correcções:
  - MLPAgent agora tem 107 inputs (+ vel_x, vel_y)
  - MoveForwardTask com stuck penalty
  - Seeds rotativas do pool [0..19] a cada geração
  - MU=20, LAMBDA=40 em DEV (mais diversidade para espaço de fitness mais rico)
  - GENERATIONS=100 em DEV

Uso:
    python evolution.py 1           # DEV  (~1h)
    python evolution.py 1 --full    # FULL (500 gens, entrega final)
"""

import sys
import time
import pickle as pkl
import csv
from collections import deque
from copy import deepcopy
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agents.mlp_agent import MLPAgent
import evaluation as eval_module
from evaluation import evaluate_population


# ---------------------------------------------------------------------------
# 1. Modo de execução
# ---------------------------------------------------------------------------
MODE = "FULL" if ("--full" in sys.argv) else "DEV"

if MODE == "DEV":
    MU            = 20    # aumentado: espaço mais rico precisa mais diversidade
    LAMBDA        = 40
    GENERATIONS   = 100
    N_TRAIN_SEEDS = 3     # sorteadas do pool a cada geração
else:
    MU            = 15
    LAMBDA        = 50
    GENERATIONS   = 500
    N_TRAIN_SEEDS = 3

eval_module.N_TRAIN_SEEDS = N_TRAIN_SEEDS
TRAIN_SEEDS = None   # None → evaluation.py sorteia do SEED_POOL a cada geração


# ---------------------------------------------------------------------------
# 2. Hiperparâmetros
# ---------------------------------------------------------------------------
N_ELITES         = 1
SIGMA_INIT       = 0.5
SIGMA_MIN        = 0.01
SIGMA_MAX        = 1.0
ADAPT_FACTOR     = 1.22
SUCCESS_TARGET   = 0.2
WINDOW           = 10
STAGNATION_LIMIT = 10
SIGMA_RESTART    = 0.3
TOURNAMENT_K     = 2
CROSSOVER_PROB   = 0.7
MUTATION_PROB    = 1.0


# ---------------------------------------------------------------------------
# 3. Utilitários
# ---------------------------------------------------------------------------
@contextmanager
def timer_context(label):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f"  [{label}] {time.perf_counter() - t0:.1f}s")


def make_evolution_plot(best_h, mean_h, sigma_h, title, save=False):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(best_h, label="Best (gen)", color="tab:blue")
    ax1.plot(mean_h, label="Mean (gen)", color="tab:orange", alpha=0.7)
    ax1.set_ylabel("Fitness")
    ax1.set_title(title)
    ax1.legend()
    ax2.plot(sigma_h, label="Sigma", color="tab:green")
    ax2.set_ylabel("Sigma")
    ax2.set_xlabel("Generation")
    ax2.legend()
    plt.tight_layout()
    if save:
        plt.savefig(f"{title}.png", dpi=150)
        print(f"  Plot: {title}.png")
    plt.close()


# ---------------------------------------------------------------------------
# 4. Operadores evolutivos
# ---------------------------------------------------------------------------
def init_individual(n):
    return (np.random.randn(n) * SIGMA_INIT).astype(np.float32)

def init_population(size, n):
    return [init_individual(n) for _ in range(size)]

def tournament_selection(pop, fits):
    idx = np.random.randint(0, len(pop), size=TOURNAMENT_K)
    return deepcopy(pop[int(idx[np.argmax(fits[idx])])])

def blx_alpha_crossover(p1, p2, alpha=0.5):
    lo  = np.minimum(p1, p2)
    hi  = np.maximum(p1, p2)
    ext = (hi - lo) * alpha
    return np.random.uniform(lo - ext, hi + ext).astype(np.float32)

def gaussian_mutation(ind, sigma):
    return ind + (np.random.randn(len(ind)) * sigma).astype(np.float32)

def make_offspring(pop, fits, n_offspring, sigma):
    offspring = []
    for _ in range(n_offspring):
        p1 = tournament_selection(pop, fits)
        p2 = tournament_selection(pop, fits)
        child = blx_alpha_crossover(p1, p2) \
                if np.random.rand() < CROSSOVER_PROB else deepcopy(p1)
        if np.random.rand() < MUTATION_PROB:
            child = gaussian_mutation(child, sigma)
        offspring.append(child)
    return offspring

def survivor_selection(parents, p_fits, offspring, o_fits, mu, elites):
    """μ+λ com elitismo explícito."""
    combined      = parents + offspring
    combined_fits = np.concatenate([p_fits, o_fits])
    order         = np.argsort(-combined_fits)
    new_pop  = [combined[i] for i in order[:mu]]
    new_fits = combined_fits[order[:mu]].copy()
    # Elitismo explícito: últimas posições reservadas
    for rank, e in enumerate(elites):
        new_pop[-(rank + 1)]  = deepcopy(e["ind"])
        new_fits[-(rank + 1)] = e["fit"]
    return new_pop, new_fits


# ---------------------------------------------------------------------------
# 5. Sigma adaptativo — 1/5 rule + restart por estagnação
# ---------------------------------------------------------------------------
def adapt_sigma(sigma, history, stagnant_gens):
    if stagnant_gens >= STAGNATION_LIMIT:
        print(f"  [SIGMA RESTART] {sigma:.4f} → {SIGMA_RESTART} "
              f"(estagnado há {stagnant_gens} gens)")
        return SIGMA_RESTART
    if len(history) < WINDOW:
        return sigma
    rate = sum(history) / len(history)
    if rate > SUCCESS_TARGET:
        return min(sigma * ADAPT_FACTOR, SIGMA_MAX)
    elif rate < SUCCESS_TARGET:
        return max(sigma / ADAPT_FACTOR, SIGMA_MIN)
    return sigma


# ---------------------------------------------------------------------------
# 6. Loop principal
# ---------------------------------------------------------------------------
def evolution_strategy(seed):
    tag = "full" if MODE == "FULL" else "dev"

    print(f"\n{'='*60}")
    print(f"  MODE={MODE} | μ={MU} | λ={LAMBDA} | GEN={GENERATIONS}")
    print(f"  N_TRAIN_SEEDS={N_TRAIN_SEEDS}/gen | pool=SEED_POOL[0..19] (rotativo)")
    print(f"  N_ELITES={N_ELITES} | σ_init={SIGMA_INIT}")
    print(f"  MLPAgent input: 107 features (+ vel_x, vel_y)")
    print(f"  MoveForwardTask: stuck penalty activa")
    print(f"{'='*60}\n")

    n_params = len(MLPAgent().get_param_vector())
    print(f"Genotype: {n_params} parameters")

    pop  = init_population(MU, n_params)
    print("Evaluating initial population...")
    fits = evaluate_population(MLPAgent, pop, None)
    print(f"Initial → best={fits.max():.2f}  mean={fits.mean():.2f}\n")

    best_idx = int(np.argmax(fits))
    best_ind = deepcopy(pop[best_idx])
    best_fit = float(fits[best_idx])
    prev_best = best_fit

    sigma         = SIGMA_INIT
    history       = deque(maxlen=WINDOW)
    stagnant_gens = 0

    best_h  = []
    mean_h  = []
    sigma_h = []

    Path("data/mlp_best_agents").mkdir(parents=True, exist_ok=True)
    csv_path = f"data/mlp_best_agents/log_{tag}_seed_{seed}.csv"

    with open(csv_path, "w", newline="") as cf:
        cw = csv.writer(cf)
        cw.writerow(["gen", "best_gen", "mean_gen", "min_gen",
                     "std_gen", "best_run", "sigma", "stagnant_gens"])

        for gen in range(GENERATIONS):
            print(f"\n--- Gen {gen+1}/{GENERATIONS} | "
                  f"σ={sigma:.4f} | stagnant={stagnant_gens} ---")

            # Elites
            elite_order = np.argsort(-fits)[:N_ELITES]
            elites = [{"ind": deepcopy(pop[i]), "fit": float(fits[i])}
                      for i in elite_order]
            print(f"  [ELITE] fit={elites[0]['fit']:.2f} → passa intacto")

            # Offspring
            offspring = make_offspring(pop, fits, LAMBDA, sigma)
            with timer_context("Eval offspring"):
                o_fits = evaluate_population(MLPAgent, offspring, None)

            # 1/5 rule
            history.append(bool(o_fits.max() > float(fits.mean())))

            # Survivor selection
            pop, fits = survivor_selection(pop, fits, offspring, o_fits,
                                           MU, elites)

            # Best-of-run
            gen_best = float(fits.max())
            if gen_best > best_fit:
                best_fit = gen_best
                best_ind = deepcopy(pop[int(np.argmax(fits))])
                fname = (f"data/mlp_best_agents/"
                         f"es_{tag}_seed_{seed}_{best_fit:.3f}.pkl")
                with open(fname, "wb") as f:
                    pkl.dump(best_ind, f)
                print(f"  >>> NEW BEST {best_fit:.3f} → {fname}")

            # Estagnação
            if gen_best > prev_best:
                stagnant_gens = 0
            else:
                stagnant_gens += 1
            prev_best = gen_best

            # Sigma
            sigma = adapt_sigma(sigma, history, stagnant_gens)
            if stagnant_gens >= STAGNATION_LIMIT:
                stagnant_gens = 0

            # Log
            mean_r = float(fits.mean())
            min_r  = float(fits.min())
            std_r  = float(fits.std())
            print(f"  best={gen_best:.2f} | mean={mean_r:.2f} | "
                  f"min={min_r:.2f} | std={std_r:.2f} | "
                  f"run_best={best_fit:.2f} | σ={sigma:.4f}")

            best_h.append(gen_best)
            mean_h.append(mean_r)
            sigma_h.append(sigma)
            cw.writerow([gen+1, gen_best, mean_r, min_r,
                         std_r, best_fit, sigma, stagnant_gens])
            cf.flush()

    make_evolution_plot(best_h, mean_h, sigma_h,
                        title=f"ES_MLP_{tag}_seed_{seed}", save=True)
    print(f"\n=== Finished. Best fitness = {best_fit:.3f} ===")
    return best_ind, best_fit


# ---------------------------------------------------------------------------
# 7. Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not positional:
        print("Usage: python evolution.py <seed> [--full]")
        sys.exit(1)

    seed = int(positional[0])
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    print(f"Seed={seed} | Mode={MODE}")
    evolution_strategy(seed)
