#!/usr/bin/python

import wandb
import cv2

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
        if not self.debug:
            if plot_img is not None:
                plot_img = cv2.resize(plot_img, (plot_img.shape[1] // 10, plot_img.shape[0] // 10))

                wandb.log({"plot_image": wandb.Image(plot_img)}, step=step)

            real_img = cv2.resize(real_img, (real_img.shape[1] // 20, real_img.shape[0] // 20))
            wandb.log({"real_image": wandb.Image(real_img)}, step=step)

    def __getattr__(self, key):
        return self.params[key]
