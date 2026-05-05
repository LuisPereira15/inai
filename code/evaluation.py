import marioai
from multiprocessing import Pool, Manager, current_process
from itertools import cycle
from agents import MLPAgent, CodeAgent
from tasks import MoveForwardTask, HunterTask
import numpy as np

# Variable that configures the number of parallel processes
N_PROCESSES = 5
# Task Definition
TASK_TO_SOLVE = MoveForwardTask


port_list = [4242 + i for i in range(N_PROCESSES)]


def evaluate_agent(agent, task, episodes=1, seeds=None):
    """
    Evaluates the agent on the task for a given number of episodes.

    For each episode, Mario attempts up to 3 difficulty levels.
    He only advances to the next difficulty if he wins on ALL 3 seeds
    at the current difficulty. Seeds are shared across the whole population
    per generation (passed in from evaluate_population), ensuring fair
    comparison between individuals.

    Returns the average cumulative reward across episodes.
    """
    if seeds is None:
        seeds = [1, 2, 3]

    exp = marioai.Experiment(task, agent)
    exp.max_fps = -1

    total_reward = 0

    for _ in range(episodes):
        episode_reward = 0
        task.level_difficulty = 0

        # Try up to 3 difficulty levels
        for _ in range(3):
            won_all_seeds = True

            # Run the agent on each of the 3 seeds at this difficulty
            for seed in seeds:
                task.env.level_seed = seed
                exp.doEpisodes(1)
                episode_reward += task.cum_reward

                if task.status != 1:  # did not WIN this seed
                    won_all_seeds = False

            # Only advance difficulty if Mario won on every seed
            if won_all_seeds:
                task.level_difficulty += 1
            else:
                break

        total_reward += episode_reward

    return total_reward / episodes


# --- GLOBAL VARIABLES FOR WORKER PROCESSES ---
worker_task = None
worker_agent = None


def init_worker(agent_class):
    """
    Runs ONCE when each worker process starts.
    Creates a persistent agent and task with a dedicated TCP connection.
    """
    global worker_agent, worker_task

    import multiprocessing
    worker_idx = int(multiprocessing.current_process().name.split('-')[-1]) - 1
    port = port_list[worker_idx % len(port_list)]

    worker_agent = agent_class()
    if worker_task is None:
        worker_task = TASK_TO_SOLVE(visualization=False, port=port, init_mario_mode=0)


def evaluate_individual(ind_info):
    """
    Runs for every individual in the population.
    Expects ind_info to be a tuple of (param_vector, seeds).
    Uses the globally cached worker_task and worker_agent.
    """
    global worker_task, worker_agent

    individual, seeds = ind_info

    # Load the new genotype into the persistent agent
    if isinstance(worker_agent, MLPAgent):
        worker_agent.set_param_vector(individual)
    elif isinstance(worker_agent, CodeAgent):
        worker_agent.action_function = individual

    try:
        reward = evaluate_agent(worker_agent, worker_task, seeds=seeds)
    except Exception as e:
        print(f"Error in worker: {e}")
        reward = 0

    return reward


def evaluate(agent_class, ind_info, seeds=None):
    global worker_agent, worker_task
    if worker_agent is None:
        worker_agent = agent_class()
    if worker_task is None:
        worker_task = TASK_TO_SOLVE(visualization=False, port=port_list[0])
    return evaluate_individual((ind_info, seeds or [1, 2, 3]))


def evaluate_population(agent, population, seeds=None):
    """
    Evaluates the entire population in parallel.

    Args:
        agent:      the agent class (e.g. MLPAgent)
        population: list of parameter vectors
        seeds:      list of 3 level seeds shared across all individuals
                    in this generation. If None, defaults to [1, 2, 3].
    """
    if seeds is None:
        seeds = [1, 2, 3]

    n_processes = N_PROCESSES

    # Pack each individual with the generation seeds so workers receive them
    population_with_seeds = [(ind, seeds) for ind in population]

    with Pool(processes=n_processes, initializer=init_worker, initargs=(agent,)) as pool:
        rewards_list = pool.map(evaluate_individual, population_with_seeds)

    return np.array(rewards_list)