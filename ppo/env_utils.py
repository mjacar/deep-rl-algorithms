# Adapted from https://github.com/openai/baselines
import numpy as np
import gym


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class Normalize(gym.Wrapper):
    """
    A wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(
        self,
        env,
        ob=True,
        ret=True,
        clipob=10.0,
        cliprew=10.0,
        gamma=0.99,
        epsilon=1e-8,
        use_tf=False,
    ):
        super(Normalize, self).__init__(env)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(1)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.ret = self.ret * self.gamma + np.array([rew])
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rew = np.clip(
                np.array([rew]) / np.sqrt(self.ret_rms.var + self.epsilon),
                -self.cliprew,
                self.cliprew,
            )
        if done:
            self.ret[0] = 0
        return obs, rew, done, info

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip(
                (obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon),
                -self.clipob,
                self.clipob,
            )
            return obs
        else:
            return obs

    def reset(self):
        self.ret[0] = 0
        obs = self.env.reset()
        return self._obfilt(obs)
