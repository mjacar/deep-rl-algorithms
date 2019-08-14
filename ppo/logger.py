import ray
import torch
import torch.utils.tensorboard as tb
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal


@ray.remote
class Logger:
    def __init__(self, env_thunk, policy_thunk, state_dict_file):
        self.writer = tb.SummaryWriter()
        self.env = env_thunk()
        self.policy = policy_thunk()
        self.state_dict_file = state_dict_file
        self.best_reward = float("-inf")

    def log_reward(self, steps, policy_state_dict, walltime):
        self.policy.load_state_dict(policy_state_dict)
        rewards, ep_lens = [], []
        for _ in range(5):
            reward, ep_len = self.simulate_episode()
            rewards.append(reward)
            ep_lens.append(ep_len)
        avg_reward = sum(rewards) / float(5)
        avg_ep_len = sum(ep_lens) / float(5)
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            torch.save(policy_state_dict, self.state_dict_file)
        self.writer.add_scalar(
            "avg_reward", avg_reward, global_step=steps, walltime=walltime
        )
        self.writer.add_scalar(
            "avg_rep_len", avg_ep_len, global_step=steps, walltime=walltime
        )

    def simulate_episode(self):
        total_reward, steps = 0, 0
        done = False
        state = self.env.reset()
        while not done:
            steps += 1
            action = self.choose_action(state)
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
        return total_reward, steps

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        mean, logstd = self.policy(state)
        dist = Independent(Normal(mean, torch.exp(logstd)), 1)
        return dist.sample().squeeze().numpy()
