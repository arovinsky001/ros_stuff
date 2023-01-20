#!/usr/bin/python

import os
import argparse
import pickle as pkl
import numpy as np
import rospy
from tqdm import trange

from mpc_agents import MPCAgent
from replay_buffer import ReplayBuffer
from logger import Logger
from train_utils import train_from_buffer
from utils import dimensions, make_state_subscriber
from environment import Environment

# seed for reproducibility
# SEED = 0
# import torch; torch.manual_seed(SEED)
# np.random.seed(SEED)
SEED = np.random.randint(0, 1e9)


class Experiment:
    def __init__(self, robot_pos, object_pos, corner_pos, robot_vel, object_vel, action_timestamp, **params):
        self.params = params

        self.agent = MPCAgent(params)
        self.replay_buffer = ReplayBuffer(params)
        self.env = Environment(robot_pos, object_pos, corner_pos, robot_vel, object_vel, action_timestamp, params, self.agent)
        # self.logger = Logger(params)

        if not self.robot_goals:
            assert self.use_object

        # pretrain agent
        self.replay_buffer.restore()
        train_from_buffer(
            self.agent, self.replay_buffer, validation_buffer=None,
            pretrain_samples=self.pretrain_samples, save_agent=self.save_agent,
            train_epochs=self.train_epochs, batch_size=self.batch_size,
            meta=self.meta,
        )

        # # load or initialize and potentially pretrain new agent
        # if load_agent:
        #     with open(AGENT_PATH, "rb") as f:
        #         self.agent = pkl.load(f)

        #     self.agent.policy.update_params_and_weights(mpc_params, cost_weights_dict)

        np.set_printoptions(suppress=True)

    def __getattr__(self, key):
        return self.params[key]

    def run(self):
        self.take_warmup_steps()

        state = self.env.reset()
        done = False
        episode = 0
        step = 0

        while not rospy.is_shutdown():
            t = rospy.get_time()
            goals = self.env.get_next_n_goals(self.agent.policy.horizon)

            action, predicted_next_state = self.agent.get_action(state, goals)

            next_state, done = self.env.step(action)

            print("TIME:", rospy.get_time() - t)
            # if predicted_next_state is not None:
            #     self.logger.log_prediction_errors(state, next_state, predicted_next_state)
            #     self.logger.log_mpc_costs(next_state, goals[0], step)

            if state and next_state:
                self.replay_buffer.add(state, action, next_state)

            if self.online:
                for _ in range(self.utd_ratio):
                    self.update_agent_online()

            state = next_state
            step += 1

            if done:
                episode += 1
                # self.prepare_next_episode()

                if episode == self.n_episodes:
                    rospy.signal_shutdown(f"Experiment finished! Did {self.n_episodes} rollouts.")

                state = self.env.reset()

    def take_warmup_steps(self):
        if self.debug:
            return

        for _ in trange(5, desc="Warmup Steps"):
            random_max_action = np.random.choice([0.999, -0.999], size=2)
            self.env.step(random_max_action)

    def define_goal_trajectory(self):
        rospy.sleep(0.2)        # wait for states to be published and set

        if self.robot_goals:
            back_circle_center_rel = np.array([0.7, 0.5])
            front_circle_center_rel = np.array([0.4, 0.5])
        else:
            back_circle_center_rel = np.array([0.65, 0.5])
            front_circle_center_rel = np.array([0.4, 0.5])

        corner_pos = self.state_dict["corner"].copy()
        self.back_circle_center = back_circle_center_rel * corner_pos[:2]
        self.front_circle_center = front_circle_center_rel * corner_pos[:2]
        self.radius = np.linalg.norm(self.back_circle_center - self.front_circle_center) / 2

    def update_agent_online(self):
        if self.replay_buffer.size > self.min_train_steps:
            for model in self.agent.models:
                states, actions, next_states = self.replay_buffer.sample(self.batch_size)

                # sample_size = min(self.total_steps, 10)
                # states, actions, next_states = self.replay_buffer.sample_recent(sample_size)
                # print(len(states), len(actions), len(next_states))

                model.update(states, actions, next_states)

    def prepare_next_episode(self):
        if np.abs(self.lap_step) == self.episode_length:
            self.laps += 1
            # Print current cumulative loss per lap completed
            dist_costs, heading_costs, total_costs = self.costs.T
            data = np.array([[dist_costs.mean(), dist_costs.std(), dist_costs.min(), dist_costs.max()],
                         [heading_costs.mean(), heading_costs.std(), heading_costs.min(), heading_costs.max()],
                         [total_costs.mean(), total_costs.std(), total_costs.min(), total_costs.max()]])
            print("lap:", self.laps)
            print("rows: (dist, heading, total)")
            print("cols: (mean, std, min, max)")
            print("DATA:", data, "\n")
            self.lap_step = 0
            self.started = False

            start_state = self.env.get_goal(step_override=0)
            self.logger.plot_states(save=True, laps=self.laps, replay_buffer=self.replay_buffer,
                                    start_state=start_state, reverse_episode=self.reverse_episode)
            self.logger.reset_plot_states()

            self.logger.plot_model_errors()
            self.logger.reset_model_errors()

            self.logger.log_performance_metrics(self.costs, self.all_actions)

            if self.online:
                self.logger.dump_agent(self.agent, self.laps, self.replay_buffer)
                self.costs = np.empty((0, self.n_costs))
            elif(not self.online and self.laps == self.n_episodes):
                self.logger.dump_agent(self.agent, self.laps, self.replay_buffer)


def main(args):
    rospy.init_node("run_experiment")

    """
    get states from state publisher
    """
    robot_pos, object_pos, corner_pos, robot_vel, object_vel, action_timestamp, tf_buffer, tf_listener = make_state_subscriber()


    """
    run experiment
    """
    # experiment = Experiment(robot_pos, object_pos, corner_pos, robot_vel, object_vel, action_timestamp, **vars(args))
    experiment = Experiment(robot_pos, object_pos, corner_pos, robot_vel, object_vel, action_timestamp, **vars(args))
    experiment.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # mpc method
    agent_subparser = parser.add_subparsers(dest='method')

    shooting_parser = agent_subparser.add_parser('shooting')
    cem_parser = agent_subparser.add_parser('cem')
    mppi_parser = agent_subparser.add_parser('mppi')

    cem_parser.add_argument('alpha', default=0.8)
    cem_parser.add_argument('n_best', default=30)
    cem_parser.add_argument('refine_iters', default=5)

    mppi_parser.add_argument('gamma', default=50)
    mppi_parser.add_argument('beta', default=0.5)
    mppi_parser.add_argument('noise_std', default=2)

    # generic
    parser.add_argument('-robot_ids', default=[0])
    parser.add_argument('-object_id', default=3)
    parser.add_argument('-use_object', default=False)

    parser.add_argument('-n_episodes', default=3)
    parser.add_argument('-tolerance', default=0.04)
    parser.add_argument('-episode_length', default=150)

    parser.add_argument('-new_buffer', default=False)
    parser.add_argument('-pretrain', default=False)
    parser.add_argument('-meta', default=False)
    parser.add_argument('-pretrain_samples', default=500)
    parser.add_argument('-min_train_steps', default=100)
    parser.add_argument('-train_epochs', default=200)

    parser.add_argument('-plot', default=False)
    parser.add_argument('-debug', default=False)

    parser.add_argument('-save_agent', default=True)
    parser.add_argument('-load_agent', default=False)

    # agent
    parser.add_argument('-ensemble_size', default=1)
    parser.add_argument('-batch_size', default=10000)
    parser.add_argument('-online', default=False)
    parser.add_argument('-utd_ratio', default=3)

    parser.add_argument('-agent_state_dim', default=3)
    parser.add_argument('-action_dim', default=2)

    parser.add_argument('-horizon', default=5)
    parser.add_argument('-mpc_samples', default=200)
    parser.add_argument('-robot_goals', default=True)

    # model
    parser.add_argument('-model_state_dim', default=4)

    parser.add_argument('-hidden_dim', default=200)
    parser.add_argument('-hidden_depth', default=1)
    parser.add_argument('-lr', default=0.001)

    parser.add_argument('-scale', default=True)
    parser.add_argument('-dist', default=False)
    parser.add_argument('-std', default=0.01)

    # replay buffer
    parser.add_argument('-save_freq', default=50)
    parser.add_argument('-capacity', default=10000)

    args = parser.parse_args()
    main(args)
