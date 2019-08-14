import ray
import time
import torch
import torch.utils.tensorboard as tb


@ray.remote
class Logger:
    def __init__(self, state_dict_file):
        self.writer = tb.SummaryWriter()
        self.step = 0
        self.best_reward = float("-inf")
        self.state_dict_file = state_dict_file

    def log_value(self, value, steps, state_dict):
        self.step += steps
        if value > self.best_reward:
            self.best_reward = value
            torch.save(state_dict, self.state_dict_file)
        self.writer.add_scalar(
            "dqn", value, global_step=self.step, walltime=time.time()
        )
