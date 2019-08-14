import torch
import numpy as np
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal


class Actor:
    def __init__(self, env, policy, value_fn, gamma, lam, max_steps_per_episode, seed):
        self.env = env
        self.env.seed(seed)
        self.state = self.env.reset()
        self.done = False
        self.max_steps_per_episode = max_steps_per_episode
        self.policy = policy
        self.value_fn = value_fn
        self.gamma = gamma
        self.lam = lam

    def generate_rollout(self):
        steps, total_reward = 0, 0
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        while steps < self.max_steps_per_episode:
            states.append(np.asarray(self.state))
            dones.append(float(self.done))
            steps += 1
            action, log_prob = self.choose_action(self.state)
            next_state, reward, self.done, _ = self.env.step(action)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            value = self.value_fn(
                torch.from_numpy(self.state).float().unsqueeze(0).cuda()
            )
            values.append(value.squeeze().item())
            if self.done:
                self.state = self.env.reset()
            else:
                self.state = next_state
        states.append(self.state)
        dones.append(float(self.done))
        value = self.value_fn(torch.from_numpy(self.state).float().unsqueeze(0).cuda())
        values.append(value.squeeze().item())
        advs, targets = self.generate_gae_targets(rewards, values, dones)
        return states[:-1], actions, advs, targets, values[:-1], log_probs

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        mean, logstd = self.policy(state.cuda())
        dist = Independent(Normal(mean.squeeze(), torch.exp(logstd)), 1)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.squeeze().cpu().numpy(), log_prob.item()

    def generate_gae_targets(self, rewards, values, dones):
        advs = [0 for _ in range(self.max_steps_per_episode)]
        current = 0
        for t in range(self.max_steps_per_episode - 1, -1, -1):
            delta = rewards[t] - values[t]
            delta += (1 - dones[t + 1]) * self.gamma * values[t + 1]
            current = delta + self.gamma * self.lam * (1 - dones[t + 1]) * current
            advs[t] = current
        targets = [advs[i] + values[i] for i in range(len(advs))]
        return advs, targets
