#!/usr/bin/python

import argparse
import numpy as np
from tqdm import trange
import rospy

from mpc_agents import RandomShootingAgent, CEMAgent, MPPIAgent
from replay_buffer import ReplayBuffer
from logger import Logger
from train_utils import train_from_buffer
from utils import make_state_subscriber
from environment import Environment

# seed for reproducibility
# SEED = 0
# import torch; torch.manual_seed(SEED)
# np.random.seed(SEED)
SEED = np.random.randint(0, 1e9)


class Experiment:
    def __init__(self, robot_pos, object_pos, corner_pos, robot_vel, object_vel, action_timestamp, **params):
        self.params = params
        params["n_robots"] = len(self.robot_ids)

        if self.mpc_method == 'mppi':
            self.agent = MPPIAgent(params)
        elif self.mpc_method == 'cem':
            self.agent = CEMAgent(params)
        elif self.mpc_method == 'shooting':
            self.agent = RandomShootingAgent(params)
        else:
            raise ValueError
        self.env = Environment(robot_pos, object_pos, corner_pos, robot_vel, object_vel, action_timestamp, params)
        self.replay_buffer = ReplayBuffer(params)
        self.replay_buffer.restore()

        if self.sample_recent_buffer:
            self.replay_buffer_sample_fn = self.replay_buffer.sample_recent
        else:
            self.replay_buffer_sample_fn = self.replay_buffer.sample

        if self.load_agent:
            self.agent.restore()
        else:
            train_from_buffer(
                self.agent, self.replay_buffer, validation_buffer=None,
                pretrain_samples=self.pretrain_samples, save_agent=self.save_agent,
                train_epochs=self.train_epochs, batch_size=self.batch_size,
                meta=self.meta,
            )

        # self.logger = Logger(params)
        np.set_printoptions(suppress=True)

    def __getattr__(self, key):
        return self.params[key]

    def run(self):
        # warmup robot before running actual experiment
        if not self.debug:
            rospy.sleep(1)
            # for _ in trange(5, desc="Warmup Steps"):
            #     random_max_action = np.random.choice([0.999, -0.999], size=2*self.n_robots)
            #     self.env.step(random_max_action)

        state = self.env.reset(self.agent, self.replay_buffer)
        # state = self.env.get_state()
        done = False
        episode = 0
        step = 0
        plot_imgs = []
        real_imgs = []

        while not rospy.is_shutdown():
            goals = self.env.get_next_n_goals(self.agent.mpc_horizon)
            action, predicted_next_state = self.agent.get_action(state, goals)
            next_state, done = self.env.step(action)

            relevant_state = state[:2] if self.robot_goals else state[-3:-1]
            relevant_pred =  predicted_next_state[:2] if self.robot_goals else predicted_next_state[-3:-1]
            relevant_next = next_state[:2] if self.robot_goals else next_state[-3:-1]
            print("\nDISTANCE FROM GOAL:", np.linalg.norm(relevant_next - goals[0, :2]))
            print("PREDICTION ERROR:", np.linalg.norm(relevant_pred - relevant_next))

            if state is not None and next_state is not None:
                self.replay_buffer.add(state, action, next_state)

            if self.update_online:
                for model in self.agent.models:
                    for _ in range(self.utd_ratio):
                        model.update(*self.replay_buffer_sample_fn(self.batch_size))

            # self.logger.log_step(state, next_state, predicted_next_state, self.env)
            # self.logger.log_images(*self.env.render(), step)

            plot_img, real_img = self.env.render()
            plot_imgs.append(plot_img)
            real_imgs.append(real_img)

            if done:
                episode += 1

                if episode == 2:
                    fps = 7

                    import cv2
                    height, width = plot_imgs[0].shape[0], plot_imgs[0].shape[1]
                    video = cv2.VideoWriter("/home/arovinsky/Desktop/plot_movie_0.avi", cv2.VideoWriter_fourcc(*'XVID'), fps, (width,height))

                    for plot_img in plot_imgs:
                        video.write(plot_img)

                    video.release()

                    div_factor = 1
                    height, width = real_imgs[0].shape[0] // div_factor, real_imgs[0].shape[1] // div_factor
                    video = cv2.VideoWriter("/home/arovinsky/Desktop/real_movie_0.avi", cv2.VideoWriter_fourcc(*'XVID'), fps, (width,height))

                    for real_img in real_imgs:
                        video.write(cv2.resize(real_img, (real_img.shape[1] // div_factor, real_img.shape[0] // div_factor)))

                    video.release()

                plot_imgs = []
                real_imgs = []

                if episode == self.n_episodes:
                    rospy.signal_shutdown(f"Experiment finished! Did {self.n_episodes} rollouts.")
                    return

                state = self.env.reset(self.agent, self.replay_buffer)
            else:
                state = next_state
                step += 1


def main(args):
    rospy.init_node("run_experiment")

    robot_pos, object_pos, corner_pos, robot_vel, object_vel, action_timestamp, tf_buffer, tf_listener = make_state_subscriber(args.robot_ids)
    experiment = Experiment(robot_pos, object_pos, corner_pos, robot_vel, object_vel, action_timestamp, **vars(args))
    experiment.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # mpc method
    # agent_subparser = parser.add_subparsers(dest='method')

    # shooting_parser = agent_subparser.add_parser('shooting')
    # cem_parser = agent_subparser.add_parser('cem')
    # mppi_parser = agent_subparser.add_parser('mppi')

    parser.add_argument('-mpc_method', type=str, default='mppi')

    parser.add_argument('-alpha', type=float, default=0.8)
    parser.add_argument('-n_best', type=int, default=30)
    parser.add_argument('-refine_iters', type=int, default=5)

    parser.add_argument('-gamma', type=int, default=50)
    parser.add_argument('-beta', type=float, default=0.5)
    parser.add_argument('-noise_std', type=float, default=2)

    # generic
    parser.add_argument('-robot_ids', nargs='+', type=int, default=[0])
    parser.add_argument('-object_id', type=int, default=3)
    parser.add_argument('-use_object', action='store_true')
    parser.add_argument('-exp_name', type=str)

    parser.add_argument('-n_episodes', type=int, default=3)
    parser.add_argument('-tolerance', type=float, default=0.04)
    parser.add_argument('-episode_length', type=int, default=150)

    parser.add_argument('-meta', action='store_true')
    parser.add_argument('-pretrain_samples', type=int, default=500)
    parser.add_argument('-train_epochs', type=int, default=200)

    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-save_agent', action='store_true')
    parser.add_argument('-load_agent', action='store_true')

    # agent
    parser.add_argument('-ensemble_size', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=10000)
    parser.add_argument('-update_online', action='store_true')
    parser.add_argument('-sample_recent_buffer', action='store_true')
    parser.add_argument('-utd_ratio', type=int, default=3)

    parser.add_argument('-mpc_horizon', type=int, default=5)
    parser.add_argument('-mpc_samples', type=int, default=200)
    parser.add_argument('-robot_goals', action='store_true')

    # model
    parser.add_argument('-hidden_dim', type=int, default=200)
    parser.add_argument('-hidden_depth', type=int, default=1)
    parser.add_argument('-lr', type=float, default=0.001)

    parser.add_argument('-scale', action='store_true')
    parser.add_argument('-dist', action='store_true')
    parser.add_argument('-std', type=bool, default=0.01)

    # replay buffer
    parser.add_argument('-save_freq', type=int, default=50) # TODO implement this
    parser.add_argument('-buffer_capacity', type=int, default=10000)

    args = parser.parse_args()
    main(args)
