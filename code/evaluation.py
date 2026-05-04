"""
evaluation.py v4

ALTERAÇÕES v4:
  - MAX_STEPS aumentado de 750 para 2000 por episódio
    (a 1.7px/step, 750 steps só percorre ~1275px de ~3000px do nível)
  - Restante lógica mantida intacta
"""

import marioai
from multiprocessing import Pool, Manager, current_process
from itertools import cycle
from agents import MLPAgent, CodeAgent
from tasks import MoveForwardTask, HunterTask
import numpy as np

# Variable that configures the number of parallel processes
N_PROCESSES = 5
# Task Definition
TASK_TO_SOLVE = MoveForwardTask  # HunterTask para Stage 2

# ALTERAÇÃO v4: aumentado de 750 para 2000
MAX_STEPS = 2000

port_list = [4242 + i for i in range(N_PROCESSES)]

def evaluate_agent(agent, task, episodes=1):
    """
    Evaluates the agent on the task for a given number of episodes.
    Returns the average fitness (reward).
    """
    exp = marioai.Experiment(task, agent)
    exp.max_fps = -1
    exp.max_steps = MAX_STEPS  # ALTERAÇÃO v4: timeout aumentado

    total_reward = 0

    for _ in range(episodes):
        episode_reward = 0
        task.level_difficulty = 0
        for _ in range(3):
            rewards = exp.doEpisodes(1)
            episode_reward += task.cum_reward

            if task.status == 1:  # WIN
                task.level_difficulty += 1
            else:
                break

        total_reward += episode_reward

    return total_reward / episodes


# --- GLOBAL VARIABLES FOR WORKER PROCESSES ---
worker_task  = None
worker_agent = None

def init_worker(agent_class):
    global worker_agent, worker_task

    import multiprocessing
    worker_idx = int(multiprocessing.current_process().name.split('-')[-1]) - 1
    port = port_list[worker_idx % len(port_list)]

    worker_agent = agent_class()
    if worker_task is None:
        worker_task = TASK_TO_SOLVE(visualization=False, port=port, init_mario_mode=0)


def evaluate_individual(ind_info):
    global worker_task, worker_agent

    if isinstance(worker_agent, MLPAgent):
        worker_agent.set_param_vector(ind_info)
    elif isinstance(worker_agent, CodeAgent):
        worker_agent.action_function = ind_info

    try:
        reward = evaluate_agent(worker_agent, worker_task)
    except Exception as e:
        print(f"Error in worker: {e}")
        reward = 0

    return reward

def evaluate(agent_class, ind_info):
    global worker_agent, worker_task
    if worker_agent is None:
        worker_agent = agent_class()
    if worker_task is None:
        worker_task = TASK_TO_SOLVE(visualization=False, port=port_list[0])
    return evaluate_individual(ind_info)


def evaluate_population(agent, population):
    n_processes = N_PROCESSES
    with Pool(processes=n_processes, initializer=init_worker, initargs=(agent,)) as pool:
        rewards_list = pool.map(evaluate_individual, population)
    worker_task = None
    return np.array(rewards_list)
