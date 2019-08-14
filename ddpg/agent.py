import argparse
import gym
import multiprocessing
import pybulletgym
import ray
import random
import torch
import numpy as np
from actor import Actor
from learner import Learner
from logger import Logger
from replay_memory import ReplayMemory
from models import Policy, ValueFunction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--env", type=str, default="HopperPyBulletEnv-v0", help="Gym environment ID"
    )
    parser.add_argument(
        "--steps_per_train", type=int, default=1, help="Steps per training iteration"
    )
    parser.add_argument(
        "--training_iters_per_update",
        type=int,
        default=10,
        help="Number of training iterations per update to the actor's copy of the model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--policy_lr",
        type=float,
        default=1e-4,
        help="Learning rate for the policy network",
    )
    parser.add_argument(
        "--value_fn_lr",
        type=float,
        default=1e-3,
        help="Learning rate for the value function network",
    )
    parser.add_argument(
        "--polyak",
        type=float,
        default=0.999,
        help="Interpolation factor in polyak averaging for target networks",
    )
    parser.add_argument(
        "--action_noise",
        type=float,
        default=0.1,
        help="Standard deviation of noise to add to actions",
    )
    parser.add_argument(
        "--l2_coeff", type=float, default=1e-2, help="L2 regularization coefficient"
    )
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
        "--replay_capacity",
        type=int,
        default=1000000,
        help="Number of transition to store in the replay buffer",
    )
    parser.add_argument(
        "--state_dict_file",
        type=str,
        default="policy_state_dict.pt",
        help="File where state dict of best policy is stored",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    env_thunk = lambda: gym.make(args.env)
    env = env_thunk()
    policy_thunk = lambda: Policy(
        env.observation_space.shape[0], env.action_space.shape[0]
    )
    value_fn_thunk = lambda: ValueFunction(
        env.observation_space.shape[0], env.action_space.shape[0]
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ray.init()
    log = Logger.remote(args.state_dict_file)
    experience_buffer = ReplayMemory.remote(args.replay_capacity)
    learn = Learner.remote(
        policy_thunk,
        value_fn_thunk,
        experience_buffer,
        args.policy_lr,
        args.value_fn_lr,
        args.gamma,
        args.polyak,
        args.learning_start,
        args.steps_per_train,
        args.batch_size,
        args.l2_coeff,
        args.max_timesteps,
    )
    n_threads = multiprocessing.cpu_count()
    actors = [
        Actor.remote(
            env_thunk,
            policy_thunk,
            learn,
            experience_buffer,
            log,
            args.steps_per_train,
            args.training_iters_per_update,
            args.learning_start // n_threads,
            args.action_noise,
            random.randint(0, 2 ** 31 - 1),
        )
        for _ in range(n_threads)
    ]
    ray.get([actor.train.remote() for actor in actors])
