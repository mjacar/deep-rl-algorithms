import random
import ray
import torch
import numpy as np


@ray.remote
class Actor:
    def __init__(
        self,
        env_thunk,
        model_thunk,
        learner,
        experience_replay,
        logger,
        steps_per_train,
        training_iters_per_update,
        seed,
    ):
        self.env = env_thunk()
        self.env.seed(seed)
        self.experience_replay = experience_replay
        self.learner = learner
        self.logger = logger
        self.steps_per_train = steps_per_train
        self.training_iters_per_update = training_iters_per_update

        self.q_network = model_thunk()
        q_state_dict, self.eps, _ = ray.get(self.learner.get_params.remote())
        self.q_network.load_state_dict(q_state_dict)

    def train(self):
        total_steps, steps_in_episode, reward_in_episode = 0, 0, 0
        state = self.env.reset()
        finished_training = False
        while not finished_training:
            total_steps += 1
            steps_in_episode += 1
            action = self.choose_action(state)
            next_state, reward, done, _ = self.env.step(action)
            reward_in_episode += reward
            self.experience_replay.push.remote(
                state, action, next_state, reward, float(done)
            )
            if done:
                state = self.env.reset()
                self.logger.log_value.remote(
                    reward_in_episode, steps_in_episode, self.q_network.state_dict()
                )
                reward_in_episode, steps_in_episode = 0, 0
            else:
                state = next_state
            if total_steps % self.steps_per_train == 0:
                self.learner.optimize.remote()
            if (
                total_steps % (self.steps_per_train * self.training_iters_per_update)
                == 0
            ):
                q_state_dict, self.eps, finished_training = ray.get(
                    self.learner.get_params.remote()
                )
                self.q_network.load_state_dict(q_state_dict)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.eps:
            return random.randint(0, self.env.action_space.n - 1)
        else:
            state = torch.from_numpy(np.asarray(state)).float() / 255.0
            return self.q_network(state.permute(2, 0, 1).unsqueeze(0)).argmax().item()
