#!/usr/bin/python

import wandb


class Logger:
    def __init__(self, params):
        self.params = params

        if not self.debug:
            wandb.init(project=self.exp_name, entity="kamigami")

    def log_step(self, cost_dict, prediction_error_dict, step):
        if not self.debug:
            wandb.log(cost_dict, step=step)
            wandb.log(prediction_error_dict, step=step)

    def __getattr__(self, key):
        return self.params[key]
