#!/usr/bin/python

import os
import argparse
import numpy as np
from tqdm import tqdm, trange
from datetime import datetime
import cv2
import pickle as pkl
import rospy

from agents import RandomShootingAgent, CEMAgent, MPPIAgent
from replay_buffer import ReplayBuffer
from train_utils import train_from_buffer
from utils import make_state_subscriber
from environment import Environment

# seed for reproducibility
# SEED = 0
# import torch; torch.manual_seed(SEED)
# np.random.seed(SEED)
SEED = np.random.randint(0, 1e9)


class Experiment:
    def __init__(self, robot_pos_dict, robot_vel_dict, object_pos, object_vel, corner_pos, action_receipt_dict, **params):
        self.params = params
        params["n_robots"] = len(self.robot_ids)
        params["robot_ids"].sort()
        params["hidden_depth"] = 4

        far_params = params.copy()
        far_params["n_robots"] = 1
        far_params["use_object"] = False
        far_params["robot_goals"] = True
        far_params["ensemble_size"] = 1

        if self.mpc_method == 'mppi':
            self.agent_class = MPPIAgent
        elif self.mpc_method == 'cem':
            self.agent_class = CEMAgent
        elif self.mpc_method == 'shooting':
            self.agent_class = RandomShootingAgent
        else:
            raise ValueError

        self.env = Environment(robot_pos_dict, robot_vel_dict, object_pos, object_vel, corner_pos, action_receipt_dict, params)

        self.close_agent = self.agent_class(params)
        self.close_replay_buffer = ReplayBuffer(params)

        if self.pretrain_close_agent:
            self.close_replay_buffer.restore(restore_path=self.close_buffer_restore_path)
            print("\n\nCLOSE BUFFER SIZE:", self.close_replay_buffer.size, "\n")
            train_from_buffer(
                self.close_agent, self.close_replay_buffer, validation_buffer=None,
                pretrain_samples=self.close_replay_buffer.size, save_agent=self.save_agent,
                train_epochs=self.train_epochs, batch_size=self.batch_size,
                meta=self.meta, close_agent=True,
            )

        # train or restore single-robot agents (far from object)
        self.far_agents = [self.agent_class(far_params) for _ in range(self.n_robots)]
        for i, agent in enumerate(self.far_agents):
            if self.load_agent:
                agent.restore(recency=self.n_robots-i)
            else:
                replay_buffer = ReplayBuffer(far_params)
                replay_buffer.restore(recency=self.n_robots-i)

                print(f"\n\nTRAINING AGENT {self.robot_ids[i]}\n")
                train_from_buffer(
                    agent, replay_buffer, validation_buffer=None,
                    pretrain_samples=self.pretrain_samples, save_agent=self.save_agent,
                    train_epochs=self.train_epochs, batch_size=self.batch_size,
                    meta=self.meta, close_agent=False,
                )
            agent.cost_weights_dict["target_heading"] = 0.

        if self.exp_name is None:
            now = datetime.now()
            self.exp_name = now.strftime("%d_%m_%Y_%H_%M_%S")

        # self.model_error_dir = os.path.expanduser(f"~/kamigami_data/model_errors/{self.exp_name}/")
        # self.distance_cost_dir = os.path.expanduser(f"~/kamigami_data/distance_costs/{self.exp_name}/")
        self.plot_video_dir = os.path.expanduser(f"~/kamigami_data/plot_videos/{self.exp_name}/")
        self.real_video_dir = os.path.expanduser(f"~/kamigami_data/real_videos/{self.exp_name}/")
        self.params_pkl_dir = os.path.expanduser(f"~/kamigami_data/params_pkls/")

        dirs = [
            # self.model_error_dir,
            # self.distance_cost_dir,
            self.plot_video_dir,
            self.real_video_dir,
            self.params_pkl_dir,
        ]

        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        np.set_printoptions(suppress=True)

    def __getattr__(self, key):
        return self.params[key]

    def run(self):
        # warmup robot before running actual experiment
        if not self.debug:
            rospy.sleep(1)
            for _ in trange(5, desc="Warmup Steps"):
                random_max_action = np.random.choice([0.999, -0.999], size=2*self.n_robots)
                self.env.step(random_max_action)

        # state = self.env.reset(self.agent, self.replay_buffer)
        state = self.env.get_state()
        done = False
        episode = 0

        plot_imgs = []
        real_imgs = []

        # model_errors = []
        # distance_costs = []

        while not rospy.is_shutdown():
            print(f"\nEPISODE STEP:", self.env.episode_step)
            t = rospy.get_time()

            if state is None:
                state = self.env.get_state()

            which_robots_close_to_object = self.which_robots_close_to_object(state)
            close_mode = np.all(which_robots_close_to_object)

            beta_param = 0.7
            object_goals = self.env.get_next_n_goals(self.mpc_horizon)
            if close_mode:
                print("ROBOTS CLOSE TO OBJECT")
                if self.close_replay_buffer.size < self.buffer_train_capacity:
                    _, predicted_next_state = self.close_agent.get_action(state, object_goals)
                    action = np.random.beta(beta_param, beta_param, size=2*self.n_robots) * 2 - 1
                else:
                    action, predicted_next_state = self.close_agent.get_action(state, object_goals)
            else:
                print("ROBOTS FAR FROM OBJECT")
                goals = np.tile(state[-3:], (self.mpc_horizon, 1))

                # make far robots come to object, close robots take random action
                action = np.random.beta(beta_param, beta_param, size=2*self.n_robots) * 2 - 1
                for i, agent in enumerate(self.far_agents):
                    if which_robots_close_to_object[i]:
                        continue

                    agent_state = state[3*i:3*(i+1)]
                    action[2*i:2*(i+1)], predicted_next_state = agent.get_action(agent_state, goals)

            next_state, done = self.env.step(action)

            if state is not None and next_state is not None:
                self.close_replay_buffer.add(state, action, next_state)

                object_next_state = next_state[-3:-1]
                print("DISTANCE FROM GOAL:", np.linalg.norm(object_next_state - object_goals[0, :2]))
                if close_mode:
                    predicted_object_state = predicted_next_state[-3:-1]
                    print("PREDICTION ERROR:", np.linalg.norm(predicted_object_state - object_next_state))

            if np.any(which_robots_close_to_object):
                    # relevant_pred = predicted_next_state[:2] if self.robot_goals else predicted_next_state[-3:-1]
                    # relevant_next = next_state[:2] if self.robot_goals else next_state[-3:-1]
                    # print("PREDICTION ERROR:", np.linalg.norm(relevant_pred - relevant_next))

                # self.buffer_train_capacity

                if self.close_replay_buffer.size == self.buffer_train_capacity:
                    train_from_buffer(
                        self.close_agent, self.close_replay_buffer, validation_buffer=None,
                        pretrain_samples=self.close_replay_buffer.size, save_agent=self.save_agent,
                        train_epochs=self.train_epochs, batch_size=self.batch_size,
                        meta=self.meta,
                    )

                if self.update_online:
                    for model in self.close_agent.models:
                        for _ in range(self.utd_ratio):
                            if self.sample_recent_buffer:
                                model.update(*self.close_replay_buffer.sample_recent(self.batch_size))
                            else:
                                model.update(*self.close_replay_buffer.sample(self.batch_size))

            # log model errors and performance costs
            # model_error = next_state - predicted_next_state
            # distance_cost = np.linalg.norm(next_state[-3:-1] - goals[0, :2])

            # model_errors.append(model_error)
            # distance_costs.append(distance_cost)
            print("TIME:", rospy.get_time() - t, "s")

            if self.record_video:
                plot_img, real_img = self.env.render(done=done, episode=episode)
                plot_imgs.append(plot_img)
                real_imgs.append(real_img)

            if done:
                if self.save_real_video:
                    log_video(np.array(self.env.camera_imgs), f"/home/arovinsky/mar1_dump/real_video_ep{episode}.avi", fps=30)

                # model_error_fname = self.model_error_dir + f"episode{episode}.npy"
                # distance_cost_fname = self.distance_cost_dir + f"episode{episode}.npy"

                # model_error_arr = np.array(model_errors)
                # distance_cost_arr = np.array(distance_costs)

                # np.save(model_error_fname, model_error_arr)
                # np.save(distance_cost_fname, distance_cost_arr)

                # if self.use_object:
                #     model_error_dist_norm = np.linalg.norm(model_error_arr[:, -3:-1], axis=1)
                # else:
                #     model_error_dist_norm = np.linalg.norm(model_error_arr[:, :2], axis=1)

                # print(f"\nMODEL ERROR MEAN: {model_error_dist_norm.mean()} || STD: {model_error_dist_norm.std()}")
                # print(f"DISTANCE ERROR MEAN: {distance_cost_arr.mean()} || STD: {distance_cost_arr.std()}")

                if self.record_video:
                    log_video(plot_imgs, self.plot_video_dir + f"plot_movie_{episode}.avi", fps=7)
                    log_video(real_imgs, self.real_video_dir + f"real_movie_{episode}.avi", fps=7)

                plot_imgs = []
                real_imgs = []

                episode += 1
                self.close_replay_buffer.dump()
                self.dump()

                if episode == self.n_episodes:
                    rospy.signal_shutdown(f"Experiment finished! Did {self.n_episodes} rollouts.")
                    return

                for _ in trange(3, desc="Warmup Steps"):
                    random_max_action = np.random.choice([0.999, -0.999], size=2*self.n_robots)
                    self.env.step(random_max_action)

                # state = self.env.reset(self.agent, self.replay_buffer)
                self.env.episode_step = 0
                self.env.reverse_episode = not self.env.reverse_episode
            else:
                state = next_state

    def which_robots_close_to_object(self, state):
        robot_states = state[:3*self.n_robots].reshape(-1, 3)
        object_state = state[-3:]
        dists_from_object = np.linalg.norm((object_state - robot_states)[:, :2], axis=1)
        return dists_from_object < self.close_radius

    def dump(self):
        # self.params["buffer_save_path"] = self.close_replay_buffer.save_path
        # self.params["buffer_restore_path"] = self.close_replay_buffer.restore_path

        with open(self.params_pkl_dir + f"{self.exp_name}.pkl", "wb") as f:
            pkl.dump(self.params, f)


def log_video(imgs, filepath, fps=7):
    height, width = imgs[0].shape[0], imgs[0].shape[1]
    video = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*'XVID'), fps, (width,height))

    for img in tqdm(imgs, desc="Saving Video"):
        video.write(img)

    video.release()

def main(args):
    print("INITIALIZING NODE")
    rospy.init_node("run_experiment")

    robot_pos_dict, robot_vel_dict, object_pos, object_vel, corner_pos, action_receipt_dict, tf_buffer, tf_listener = make_state_subscriber(args.robot_ids)
    experiment = Experiment(robot_pos_dict, robot_vel_dict, object_pos, object_vel, corner_pos, action_receipt_dict, **vars(args))
    experiment.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-mpc_method', type=str, default='mppi')
    parser.add_argument('-trajectory', type=str, default='8')

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
    parser.add_argument('-exp_name', type=str, default=None)

    parser.add_argument('-n_episodes', type=int, default=3)
    parser.add_argument('-tolerance', type=float, default=0.04)
    parser.add_argument('-episode_length', type=int, default=150)

    parser.add_argument('-meta', action='store_true')
    parser.add_argument('-pretrain_samples', type=int, default=500)
    parser.add_argument('-train_epochs', type=int, default=200)

    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-save_agent', action='store_true')
    parser.add_argument('-load_agent', action='store_true')
    parser.add_argument('-record_video', action='store_true')
    parser.add_argument('-save_real_video', action='store_true')
    parser.add_argument('-random_data', action='store_true')
    parser.add_argument('-save_states', action='store_true')

    # agent
    parser.add_argument('-ensemble_size', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=10000)
    parser.add_argument('-update_online', action='store_true')
    parser.add_argument('-sample_recent_buffer', action='store_true')
    parser.add_argument('-utd_ratio', type=int, default=3)
    parser.add_argument('-discount_factor', type=float, default=0.9)

    parser.add_argument('-mpc_horizon', type=int, default=5)
    parser.add_argument('-mpc_samples', type=int, default=200)
    parser.add_argument('-robot_goals', action='store_true')
    parser.add_argument('-close_radius', type=float, default=0.4)
    parser.add_argument('-pretrain_close_agent', action='store_true')

    # model
    parser.add_argument('-hidden_dim', type=int, default=200)
    parser.add_argument('-hidden_depth', type=int, default=1)
    parser.add_argument('-lr', type=float, default=0.001)

    parser.add_argument('-scale', action='store_true')
    parser.add_argument('-dist', action='store_true')
    parser.add_argument('-std', type=float, default=0.01)

    # replay buffer
    parser.add_argument('-save_freq', type=int, default=50) # TODO implement this
    parser.add_argument('-buffer_capacity', type=int, default=10000)
    parser.add_argument('-buffer_train_capacity', type=int, default=100)
    parser.add_argument('-buffer_save_dir', type=str, default='~/kamigami_data/replay_buffers/online_buffers/')
    parser.add_argument('-buffer_restore_dir', type=str, default='~/kamigami_data/replay_buffers/meshgrid_buffers/')
    parser.add_argument('-close_buffer_restore_path', type=str, default='~/kamigami_data/replay_buffers/online_buffers/2object_tetherless_0615_duration3.npz')
    # parser.add_argument('-close_buffer_restore_path', type=str, default='~/kamigami_data/replay_buffers/online_buffers/2object_tetherless.npz')
    # parser.add_argument('-close_buffer_restore_path', type=str, default='/home/arovinsky/kamigami_data/replay_buffers/random_buffers/2object_beta300.npz')


    args = parser.parse_args()
    main(args)
