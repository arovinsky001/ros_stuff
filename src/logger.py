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

    def log_images(self, plot_img, real_img, step):
        wandb.log({"plot_image": wandb.Image(plot_img)}, step=step)
        wandb.log({"real_image": wandb.Image(real_img)}, step=step)

    def __getattr__(self, key):
        return self.params[key]
