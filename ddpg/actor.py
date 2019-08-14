import ray
import torch


@ray.remote
class Actor:
    def __init__(
        self,
        env_thunk,
        policy_thunk,
        learner,
        experience_replay,
        logger,
        steps_per_train,
        training_iters_per_update,
        learning_start,
        action_noise,
        seed,
    ):
        self.env = env_thunk()
        self.env.seed(seed)
        self.experience_replay = experience_replay
        self.learner = learner
        self.logger = logger
        self.steps_per_train = steps_per_train
        self.training_iters_per_update = training_iters_per_update
        self.learning_start = learning_start
        self.action_noise = action_noise

        self.policy_network = policy_thunk()
        policy_state_dict, _ = ray.get(self.learner.get_params.remote())
        self.policy_network.load_state_dict(policy_state_dict)

    def train(self):
        total_steps, steps_in_episode, reward_in_episode = 0, 0, 0
        state = self.env.reset()
        finished_training = False
        while not finished_training:
            total_steps += 1
            steps_in_episode += 1
            action = self.choose_action(state, total_steps)
            next_state, reward, done, _ = self.env.step(action)
            reward_in_episode += reward
            self.experience_replay.push.remote(
                state, action, next_state, reward, float(done)
            )
            if done:
                state = self.env.reset()
                self.logger.log_value.remote(
                    reward_in_episode,
                    steps_in_episode,
                    self.policy_network.state_dict(),
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
                policy_state_dict, finished_training = ray.get(
                    self.learner.get_params.remote()
                )
                self.policy_network.load_state_dict(policy_state_dict)

    def choose_action(self, state, steps):
        if steps < self.learning_start:
            return self.env.action_space.sample()
        else:
            state = torch.from_numpy(state).unsqueeze(0)
            action = self.policy_network(state).squeeze().detach()
            return (self.action_noise * torch.randn_like(action) + action).numpy()
