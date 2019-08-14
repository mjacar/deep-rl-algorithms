import ray
import torch
import torch.optim as optim
from torch.nn.modules.loss import MSELoss
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector


@ray.remote(num_gpus=1)
class Learner:
    def __init__(
        self,
        policy_thunk,
        value_fn_thunk,
        experience_replay,
        policy_lr,
        value_fn_lr,
        gamma,
        polyak,
        learning_start,
        steps_per_train,
        batch_size,
        l2_coeff,
        max_timesteps,
    ):
        self.online_policy = policy_thunk()
        self.online_policy.cuda()
        self.target_policy = policy_thunk()
        self.target_policy.cuda()
        self.target_policy.load_state_dict(self.online_policy.state_dict())

        self.online_value_fn = value_fn_thunk()
        self.online_value_fn.cuda()
        self.target_value_fn = value_fn_thunk()
        self.target_value_fn.cuda()
        self.target_value_fn.load_state_dict(self.online_value_fn.state_dict())

        self.experience_replay = experience_replay
        self.gamma = gamma
        self.polyak = polyak
        self.total_steps = 0

        self.policy_opt = optim.Adam(
            self.online_policy.parameters(), policy_lr, weight_decay=l2_coeff
        )
        self.value_fn_opt = optim.Adam(
            self.online_value_fn.parameters(), value_fn_lr, weight_decay=l2_coeff
        )
        self.value_fn_criterion = MSELoss()
        self.learning_start = learning_start
        self.steps_per_train = steps_per_train
        self.batch_size = batch_size
        self.max_timesteps = max_timesteps

    def optimize(self):
        self.total_steps += self.steps_per_train

        if self.total_steps >= self.learning_start:
            experience_sample = ray.get(
                self.experience_replay.sample.remote(self.batch_size)
            )
            state = torch.cat(
                [
                    torch.from_numpy(s.state).cuda().unsqueeze(0)
                    for s in experience_sample
                ]
            )
            next_state = torch.cat(
                [
                    torch.from_numpy(s.next_state).cuda().unsqueeze(0)
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

            # Train value function
            target = (
                reward
                + self.gamma
                * (1 - terminal)
                * self.target_value_fn(next_state, self.target_policy(next_state))
            ).detach()
            actual = self.online_value_fn(state, action)
            value_fn_loss = self.value_fn_criterion(target, actual)
            value_fn_loss.backward()
            self.value_fn_opt.step()
            self.online_policy.zero_grad()
            self.online_value_fn.zero_grad()

            # Train policy
            policy_loss = -self.online_value_fn(state, self.online_policy(state)).mean()
            policy_loss.backward()
            self.policy_opt.step()
            self.online_policy.zero_grad()
            self.online_value_fn.zero_grad()

            # Update target networks
            v_policy = parameters_to_vector(self.online_policy.parameters())
            v_policy_targ = parameters_to_vector(self.target_policy.parameters())
            new_v_policy_targ = (
                self.polyak * v_policy_targ + (1 - self.polyak) * v_policy
            )
            vector_to_parameters(new_v_policy_targ, self.target_policy.parameters())

            v_value_fn = parameters_to_vector(self.online_value_fn.parameters())
            v_value_fn_targ = parameters_to_vector(self.target_value_fn.parameters())
            new_v_value_fn_targ = (
                self.polyak * v_value_fn_targ + (1 - self.polyak) * v_value_fn
            )
            vector_to_parameters(new_v_value_fn_targ, self.target_value_fn.parameters())

    def get_params(self):
        done = self.total_steps >= self.max_timesteps
        self.online_policy.cpu()
        sd = self.online_policy.state_dict()
        self.online_policy.cuda()
        return sd, done
