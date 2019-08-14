import argparse
import multiprocessing
import random
import ray
import torch
import numpy as np
from actor import Actor
from env_utils import make_atari
from learner import Learner
from logger import Logger
from model import DQN
from replay_memory import ReplayMemory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--env", type=str, default="PongNoFrameskip-v4", help="Gym environment ID"
    )
    parser.add_argument(
        "--steps_per_train", type=int, default=4, help="Steps per training iteration"
    )
    parser.add_argument(
        "--training_iters_per_update",
        type=int,
        default=10,
        help="Number of training iterations per update to the actor's copy of the model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--steps_per_target_update",
        type=int,
        default=8000,
        help="Steps between updates to the target network",
    )
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument(
        "--learning_start",
        type=int,
        default=50000,
        help="Number of steps to take before starting to learn",
    )
    parser.add_argument(
        "--max_timesteps",
        type=int,
        default=2000000,
        help="Total number of timesteps to train for",
    )
    parser.add_argument(
        "--final_eps",
        type=float,
        default=0.1,
        help="Final epsilon for epsilon-greedy policy",
    )
    parser.add_argument(
        "--exploration_fraction",
        type=float,
        default=0.5,
        help="Percent of max_timesteps over which to linearly anneal epsilon from 1.0 to final_eps",
    )
    parser.add_argument(
        "--replay_capacity",
        type=int,
        default=200000,
        help="Number of transitions to store in the replay buffer",
    )
    parser.add_argument(
        "--adam_eps",
        type=float,
        default=1.5e-4,
        help="Numerical stability term for Adam optimization",
    )
    parser.add_argument(
        "--state_dict_file",
        type=str,
        default="q_network_state_dict.pt",
        help="File where state dict of best model is stored",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    env_thunk = lambda: make_atari(args.env)
    model_thunk = lambda: DQN()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ray.init()
    log = Logger.remote(args.state_dict_file)
    experience_buffer = ReplayMemory.remote(args.replay_capacity)
    learn = Learner.remote(
        model_thunk,
        experience_buffer,
        args.lr,
        args.gamma,
        args.learning_start,
        args.steps_per_train,
        args.batch_size,
        args.max_timesteps,
        args.final_eps,
        args.exploration_fraction,
        args.steps_per_target_update,
        args.adam_eps,
    )
    n_threads = multiprocessing.cpu_count()
    actors = [
        Actor.remote(
            env_thunk,
            model_thunk,
            learn,
            experience_buffer,
            log,
            args.steps_per_train,
            args.training_iters_per_update,
            random.randint(0, 2 ** 31 - 1),
        )
        for _ in range(n_threads)
    ]
    ray.get([actor.train.remote() for actor in actors])
