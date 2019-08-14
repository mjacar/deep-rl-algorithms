import ray
import numpy as np
import torch
import torch.optim as optim
from torch.nn.modules.loss import MSELoss


@ray.remote(num_gpus=1)
class Learner:
    def __init__(
        self,
        model_thunk,
        experience_replay,
        lr,
        gamma,
        learning_start,
        steps_per_train,
        batch_size,
        max_timesteps,
        final_eps,
        exploration_fraction,
        steps_per_target_update,
        adam_eps,
    ):
        self.online_network = model_thunk()
        self.online_network.cuda()
        self.target_network = model_thunk()
        self.target_network.cuda()
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.experience_replay = experience_replay
        self.eps = 1.0
        self.gamma = gamma
        self.total_steps = 0

        self.opt = optim.Adam(self.online_network.parameters(), lr, eps=adam_eps)
        self.criterion = MSELoss()
        self.learning_start = learning_start
        self.steps_per_train = steps_per_train
        self.batch_size = batch_size
        self.max_timesteps = max_timesteps
        self.final_eps = final_eps
        self.exploration_fraction = exploration_fraction
        self.steps_per_target_update = steps_per_target_update

    def optimize(self):
        self.total_steps += self.steps_per_train

        if self.total_steps >= self.learning_start:
            experience_sample = ray.get(
                self.experience_replay.sample.remote(self.batch_size)
            )
            state = torch.cat(
                [
                    (torch.from_numpy(np.asarray(s.state)).float() / 255.0)
                    .cuda()
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    for s in experience_sample
                ]
            )
            next_state = torch.cat(
                [
                    (torch.from_numpy(np.asarray(s.next_state)).float() / 255.0)
                    .cuda()
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    for s in experience_sample
                ]
            )
            terminal = (
                torch.tensor([s.terminal for s in experience_sample])
                .cuda()
                .unsqueeze(1)
            )
            reward = (
                torch.tensor([s.reward for s in experience_sample]).cuda().unsqueeze(1)
            )
            action = torch.tensor([s.action for s in experience_sample]).cuda()

            target = (
                reward
                + self.gamma
                * (1 - terminal)
                * self.target_network(next_state).max(1)[0].unsqueeze(1)
            ).detach()
            actual = self.online_network(state).gather(1, action.unsqueeze(1))
            loss = self.criterion(target, actual)
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

        if self.total_steps % self.steps_per_target_update == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

    def get_params(self):
        if self.total_steps < self.max_timesteps * self.exploration_fraction:
            frac = self.total_steps / float(
                self.max_timesteps * self.exploration_fraction
            )
            eps = (self.final_eps - 1.0) * frac + 1.0
        else:
            eps = self.final_eps
        done = self.total_steps >= self.max_timesteps

        self.online_network.cpu()
        sd = self.online_network.state_dict()
        self.online_network.cuda()
        return sd, eps, done
