"""
evaluation.py — Motor de avaliação paralela para a ES do Mario.

Mudanças principais vs. versão anterior:
  - Multi-seed evaluation: cada indivíduo é avaliado em N_TRAIN_SEEDS níveis
    distintos (seeds fixas, iguais para todos os indivíduos) e a fitness
    final é a MÉDIA. Reduz drasticamente o overfitting a um único layout.
  - Sem curriculum durante o treino: o curriculum (subir difficulty se WIN)
    gerava ruído porque desperdiçava avaliações em níveis difíceis antes de
    o agente saber completar os fáceis. Volta na Phase 2.
  - N_PROCESSES e porta base configuráveis: em DEV pode querer menos workers.
"""

import marioai
from multiprocessing import Pool
from agents import MLPAgent, CodeAgent
from tasks import MoveForwardTask, HunterTask
import numpy as np

# ---------------------------------------------------------------------------
# Configuração global
# ---------------------------------------------------------------------------
N_PROCESSES   = 5                     # workers paralelos (= portas docker)
TASK_TO_SOLVE = MoveForwardTask       # trocar para HunterTask na Stage 2

# Sementes de treino: FIXAS e iguais para todos os indivíduos.
# Escolhidas para cobrir layouts variados em difficulty=0.
# Em DEV usam-se as 2 primeiras; em FULL as 3 primeiras.
# O evolution.py controla qual subconjunto usar via TRAIN_SEEDS.
# Pool de 20 seeds de treino.
# A cada geração sorteia-se N_TRAIN_SEEDS deste pool SEM repetição.
# Evita memorização de layouts → força generalização real.
SEED_POOL = list(range(0, 20))
ALL_TRAIN_SEEDS = SEED_POOL   # compatibilidade com código existente

# N_TRAIN_SEEDS é lido do evolution.py em runtime via eval_module.N_TRAIN_SEEDS
N_TRAIN_SEEDS = 3

port_list = [4242 + i for i in range(N_PROCESSES)]


# ---------------------------------------------------------------------------
# Avaliação de um único episódio
# ---------------------------------------------------------------------------
def _run_one_train_episode(agent, task, exp, level_seed: int) -> float:
    """
    Corre um episódio em difficulty=0, level_seed fixo.
    Devolve task.cum_reward como fitness do episódio.
    """
    task.level_difficulty  = 0
    task.env.level_type    = 0
    task.env.level_seed    = level_seed
    exp.doEpisodes(1)
    return task.cum_reward


def evaluate_agent(agent, task, train_seeds=None) -> float:
    """
    Avalia o agente em cada seed de treino e devolve a média.

    Não há curriculum aqui — cada episódio é sempre difficulty=0.
    Isso dá um sinal de fitness estável e comparável entre indivíduos.
    """
    if train_seeds is None:
        train_seeds = list(np.random.choice(SEED_POOL, N_TRAIN_SEEDS, replace=False))

    exp = marioai.Experiment(task, agent)
    exp.max_fps = -1   # máxima velocidade durante treino

    total = 0.0
    for seed in train_seeds:
        total += _run_one_train_episode(agent, task, exp, seed)

    return total / len(train_seeds)


# ---------------------------------------------------------------------------
# Variáveis globais dos workers (uma por processo)
# ---------------------------------------------------------------------------
worker_task  = None
worker_agent = None
worker_seeds = None   # subconjunto de ALL_TRAIN_SEEDS para este worker


def init_worker(agent_class, train_seeds=None):
    """
    Inicialização do processo worker — corre UMA VEZ por worker.
    Cria a ligação ao servidor Mario e o agente persistente.
    """
    global worker_agent, worker_task, worker_seeds
    import multiprocessing

    worker_seeds = train_seeds if train_seeds is not None \
                   else list(np.random.choice(SEED_POOL, N_TRAIN_SEEDS, replace=False))

    worker_idx = int(multiprocessing.current_process().name.split('-')[-1]) - 1
    port = port_list[worker_idx % len(port_list)]

    worker_agent = agent_class()
    worker_task  = TASK_TO_SOLVE(visualization=False, port=port,
                                 init_mario_mode=0)


def evaluate_individual(ind_info) -> float:
    """
    Avalia um único indivíduo — chamado pelo pool.map para cada membro
    da população. Usa a task e o agente globais do worker.
    """
    global worker_task, worker_agent, worker_seeds

    # Atualizar o agente com o genotipo
    if isinstance(worker_agent, MLPAgent):
        worker_agent.set_param_vector(ind_info)
    elif isinstance(worker_agent, CodeAgent):
        worker_agent.action_function = ind_info

    try:
        reward = evaluate_agent(worker_agent, worker_task, worker_seeds)
    except Exception as e:
        print(f"[Worker error] {e}")
        reward = 0.0

    return reward


# ---------------------------------------------------------------------------
# Interface pública: avalia toda a população em paralelo
# ---------------------------------------------------------------------------
def evaluate_population(agent_class, population, train_seeds=None) -> np.ndarray:
    """
    Avalia toda a população em paralelo com N_PROCESSES workers.

    Parameters
    ----------
    agent_class  : classe do agente (MLPAgent ou CodeAgent)
    population   : lista de vetores de parâmetros (genótipos)
    train_seeds  : subconjunto de seeds a usar (None → ALL_TRAIN_SEEDS[:N_TRAIN_SEEDS])

    Returns
    -------
    np.ndarray de fitness com len == len(population)
    """
    # Se não passadas seeds explícitas → sortear N_TRAIN_SEEDS do SEED_POOL
    if train_seeds is None:
        seeds = list(np.random.choice(SEED_POOL, N_TRAIN_SEEDS, replace=False))
        print(f"  [Seeds desta geração] {seeds}")
    else:
        seeds = train_seeds

    with Pool(
        processes=N_PROCESSES,
        initializer=init_worker,
        initargs=(agent_class, seeds)
    ) as pool:
        rewards_list = pool.map(evaluate_individual, population)

    return np.array(rewards_list)


# ---------------------------------------------------------------------------
# Utilitário standalone (para testes fora da ES)
# ---------------------------------------------------------------------------
def evaluate(agent_class, ind_info, train_seeds=None) -> float:
    """Avalia um único indivíduo sem multiprocessing — útil para debug."""
    global worker_agent, worker_task, worker_seeds

    seeds = train_seeds if train_seeds is not None \
            else list(np.random.choice(SEED_POOL, N_TRAIN_SEEDS, replace=False))

    if worker_agent is None:
        worker_agent = agent_class()
    if worker_task is None:
        worker_task = TASK_TO_SOLVE(visualization=False,
                                    port=port_list[0],
                                    init_mario_mode=0)
    worker_seeds = seeds

    if isinstance(worker_agent, MLPAgent):
        worker_agent.set_param_vector(ind_info)
    elif isinstance(worker_agent, CodeAgent):
        worker_agent.action_function = ind_info

    return evaluate_agent(worker_agent, worker_task, seeds)
