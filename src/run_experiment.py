#!/usr/bin/python

import argparse
import numpy as np
from tqdm import trange
import rospy

from mpc_agents import MPCAgent
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

        self.agent = MPCAgent(params)
        self.env = Environment(robot_pos, object_pos, corner_pos, robot_vel, object_vel, action_timestamp, params, self.agent)
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

        self.logger = Logger(params)
        np.set_printoptions(suppress=True)

    def __getattr__(self, key):
        return self.params[key]

    def run(self):
        # warmup robot before running actual experiment
        if not self.debug:
            for _ in trange(5, desc="Warmup Steps"):
                random_max_action = np.random.choice([0.999, -0.999], size=2)
                self.env.step(random_max_action)

        state = self.env.reset()
        done = False
        episode = 0
        step = 0

        while not rospy.is_shutdown():
            goals = self.env.get_next_n_goals(self.agent.policy.mpc_horizon)
            action, predicted_next_state = self.agent.get_action(state, goals)
            next_state, done = self.env.step(action)

            if state and next_state:
                self.replay_buffer.add(state, action, next_state)

            if self.update_online:
                for model in self.agent.models:
                    for _ in range(self.utd_ratio):
                        model.update(*self.replay_buffer_sample_fn(self.batch_size))

            self.logger.log_step(state, next_state, predicted_next_state, self.env)

            if done:
                episode += 1

                if episode == self.n_episodes:
                    rospy.signal_shutdown(f"Experiment finished! Did {self.n_episodes} rollouts.")
                    return

                state = self.env.reset()
            else:
                state = next_state
                step += 1


def main(args):
    rospy.init_node("run_experiment")

    robot_pos, object_pos, corner_pos, robot_vel, object_vel, action_timestamp, tf_buffer, tf_listener = make_state_subscriber()
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
    parser.add_argument('-robot_ids', nargs='+', default=[0])
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
    parser.add_argument('-update_online', default=False)
    parser.add_argument('-sample_recent_buffer', default=False)
    parser.add_argument('-utd_ratio', default=3)

    parser.add_argument('-agent_state_dim', default=3)
    parser.add_argument('-action_dim', default=2)

    parser.add_argument('-mpc_horizon', default=5)
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
