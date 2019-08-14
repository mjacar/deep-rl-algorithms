import argparse
import gym
import math
import pybulletgym
import random
import ray
import time
import torch
import numpy as np
from actor import Actor
from env_utils import Normalize
from logger import Logger
from models import Policy, ValueFunction
from torch.optim import Adam
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--env", type=str, default="HopperPyBulletEnv-v0", help="Gym environment ID"
    )
    parser.add_argument(
        "--steps_per_rollout", type=int, default=2048, help="Steps per rollout"
    )
    parser.add_argument(
        "--batches_per_epoch", type=int, default=32, help="Number of batches per epoch"
    )
    parser.add_argument(
        "--epochs_per_update", type=int, default=10, help="Number of epochs per update"
    )
    parser.add_argument(
        "--clip_range",
        type=float,
        default=0.2,
        help="Initial value for hyperparameter for clipping in the policy objective",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--lam",
        type=float,
        default=0.95,
        help="Coefficient for generalized advantage estimation",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Initial learning rate for the policy and value function network",
    )
    parser.add_argument(
        "--entropy_coeff",
        type=float,
        default=0.00,
        help="Coefficient for weighting entropy regularization in the total loss",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.5,
        help="Coefficient for gradient norm clipping",
    )
    parser.add_argument(
        "--max_timesteps",
        type=int,
        default=1000000,
        help="Total number of timesteps to train for",
    )
    parser.add_argument(
        "--state_dict_file",
        type=str,
        default="policy_state_dict.pt",
        help="File where state dict of best policy is stored",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    env = Normalize(gym.make(args.env))
    test_env_thunk = lambda: Normalize(gym.make(args.env), ret=False)
    policy_thunk = lambda: Policy(
        env.observation_space.shape[0], env.action_space.shape[0]
    )
    value_fn = ValueFunction(env.observation_space.shape[0])
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ray.init()
    log = Logger.remote(test_env_thunk, policy_thunk, args.state_dict_file)
    policy = policy_thunk()
    policy.cuda()
    value_fn.cuda()
    actor = Actor(
        env,
        policy,
        value_fn,
        args.gamma,
        args.lam,
        args.steps_per_rollout,
        random.randint(0, 2 ** 31 - 1),
    )

    steps = 0
    update = 0
    while steps < args.max_timesteps:
        # Miscellaneous bookkeeping
        steps += args.steps_per_rollout
        update += 1

        frac = steps / float(args.max_timesteps)
        alpha = -1.0 * frac + 1.0
        policy_opt = Adam(policy.parameters(), alpha * args.lr)
        value_fn_opt = Adam(value_fn.parameters(), alpha * args.lr)

        policy.cpu()
        policy_state_dict = policy.state_dict()
        policy.cuda()
        log.log_reward.remote(steps, policy_state_dict, time.time())

        # Generate rollout
        rollout = actor.generate_rollout()
        combined_rollout = []
        for j in range(len(rollout[0])):
            state = rollout[0][j]
            action = rollout[1][j]
            adv = rollout[2][j]
            target = rollout[3][j]
            value = rollout[4][j]
            log_prob = rollout[5][j]
            combined_rollout.append((state, action, adv, target, value, log_prob))

        # Optimize policy
        random.shuffle(combined_rollout)
        batch_size = len(combined_rollout) // args.batches_per_epoch
        for _ in range(args.epochs_per_update):
            for batch_num in range(args.batches_per_epoch):
                rollout_batch = combined_rollout[
                    batch_num * batch_size : (batch_num + 1) * batch_size
                ]
                states = torch.cat(
                    [
                        torch.from_numpy(sample[0]).float().unsqueeze(0)
                        for sample in rollout_batch
                    ]
                ).cuda()
                actions = torch.cat(
                    [
                        torch.from_numpy(sample[1]).float().unsqueeze(0)
                        for sample in rollout_batch
                    ]
                ).cuda()
                advs = torch.cat(
                    [
                        torch.tensor(sample[2]).float().unsqueeze(0)
                        for sample in rollout_batch
                    ]
                ).cuda()
                targets = torch.cat(
                    [
                        torch.tensor(sample[3]).float().unsqueeze(0)
                        for sample in rollout_batch
                    ]
                ).cuda()
                old_values = torch.cat(
                    [torch.tensor(sample[4]).unsqueeze(0) for sample in rollout_batch]
                ).cuda()
                old_log_probs = torch.cat(
                    [torch.tensor(sample[5]).unsqueeze(0) for sample in rollout_batch]
                ).cuda()

                means, logstd = policy(states)
                dist = Independent(
                    Normal(
                        means, torch.exp(logstd.unsqueeze(0).expand(batch_size, -1))
                    ),
                    1,
                )
                log_probs = dist.log_prob(actions.squeeze())

                values = value_fn(states)
                clipped_values = old_values + torch.clamp(
                    values - old_values, -args.clip_range, args.clip_range
                )
                l_vf1 = (values - targets).pow(2)
                l_vf2 = (clipped_values - targets).pow(2)
                value_fn_loss = 0.5 * torch.max(l_vf1, l_vf2).mean()
                value_fn_loss.backward()
                clip_grad_norm_(value_fn.parameters(), args.max_grad_norm)
                value_fn_opt.step()
                value_fn_opt.zero_grad()

                k = logstd.shape[0]
                entropy = (k / 2) * (1 + math.log(2 * math.pi)) + 0.5 * torch.log(
                    torch.exp(logstd).pow(2).prod()
                )
                advs = ((advs - advs.mean()) / (advs.std() + 1e-8)).squeeze()

                prob_ratio = torch.exp(log_probs - old_log_probs)
                clipped_ratio = torch.clamp(
                    prob_ratio, 1 - args.clip_range, 1 + args.clip_range
                )
                l_cpi = prob_ratio * advs
                l_clip = clipped_ratio * advs
                policy_loss = -torch.min(l_cpi, l_clip).mean()

                policy_loss = policy_loss - args.entropy_coeff * entropy
                policy_loss.backward()
                clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                policy_opt.step()
                policy_opt.zero_grad()
