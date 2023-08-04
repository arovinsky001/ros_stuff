#!/usr/bin/python

import argparse
import numpy as np
import rospy
from tqdm import tqdm, trange

from environment import Environment
from replay_buffer import ReplayBuffer
from utils import make_state_subscriber


class DataCollector:
    def __init__(self, robot_pos_dict, robot_vel_dict, object_pos, object_vel, corner_pos, action_receipt_dict, **params):
        self.params = params
        params["episode_length"] = np.inf
        params["robot_ids"].sort()
        params["n_robots"] = len(self.robot_ids)
        # params["robot_goals"] = False

        if self.policy:
            dir = 'policy_buffers'
        elif self.random_data:
            dir = 'random_buffers'
        else:
            dir = 'meshgrid_buffers'

        params["buffer_save_dir"] = f"~/kamigami_data/replay_buffers/{dir}/"

        # states
        self.env = Environment(robot_pos_dict, robot_vel_dict, object_pos, object_vel, corner_pos, action_receipt_dict, params, precollecting=True)
        self.replay_buffer = ReplayBuffer(params, precollecting=True)

        action_range = np.linspace(-1, 1, np.floor(np.sqrt(self.n_samples)).astype("int"))
        left_actions, right_actions = np.meshgrid(action_range, action_range)
        self.fixed_actions = np.stack((left_actions, right_actions)).transpose(2, 1, 0).reshape(-1, 2)

        if self.policy:
            from agents import MPPIAgent
            from train_utils import train_from_buffer

            far_params = params.copy()
            far_params["n_robots"] = 1
            far_params["use_object"] = False
            # far_params["robot_goals"] = True
            far_params["ensemble_size"] = 1
            far_params["buffer_restore_dir"] = "~/kamigami_data/replay_buffers/meshgrid_buffers/"

            self.far_agents = [MPPIAgent(far_params) for _ in range(self.n_robots)]
            for i, agent in enumerate(self.far_agents):
                if self.load_agent:
                    agent.restore(recency=self.n_robots-i)
                else:
                    replay_buffer = ReplayBuffer(far_params)
                    replay_buffer.restore(recency=self.n_robots-i)

                    print(f"\n\nTRAINING AGENT {self.robot_ids[i]}\n")
                    train_from_buffer(
                        agent, replay_buffer, validation_buffer=None,
                        pretrain_samples=36, save_agent=False,
                        train_epochs=100, batch_size=100,
                        meta=False, close_agent=False,
                    )
                agent.cost_weights_dict["target_heading"] = 0.

    def __getattr__(self, key):
        return self.params[key]

    def run(self):
        # self.take_warmup_steps()

        rospy.sleep(1.)
        state = self.env.get_state()

        print("\nCOLLECTING DATA\n")

        for i in trange(self.n_samples):
            valid = False

            while not valid:
                beta_param = 0.9
                if self.policy:
                    if i % 10 < 6:
                        action = np.random.beta(beta_param, beta_param, size=2*self.n_robots) * 2 - 1
                    else:
                        action = np.empty(2*self.n_robots)
                        goal = np.tile(state[-3:], (self.mpc_horizon, 1))
                        for ii, agent in enumerate(self.far_agents):
                            goal_rand = goal + np.random.uniform(low=-0.12, high=0.12, size=goal.shape)
                            action[ii*2:(ii+1)*2], _ = agent.get_action(state[ii*3:(ii+1)*3], goal_rand)
                elif self.random_data:
                    if self.beta:
                        beta_param = 0.7
                        action = np.random.beta(beta_param, beta_param, size=2*self.n_robots) * 2 - 1
                    else:
                        action = np.random.uniform(low=-1., high=1., size=2*self.n_robots)
                else:
                    action = self.fixed_actions[i]

                next_state, _ = self.env.step(action)
                tqdm.write(str(action))

                if state is not None and next_state is not None:
                    self.replay_buffer.add(state, action, next_state)
                    valid = True

                    if self.policy and i != 0 and i % 20 == 0:
                        import pdb;pdb.set_trace()
                        state = self.env.get_state()
                    else:
                        state = next_state
                else:
                    state = self.env.get_state()

        print("\nDATA COLLECTED, SAVING REPLAY BUFFER\n")
        self.replay_buffer.dump()
        print("\nREPLAY BUFFER SAVED\n")

    def take_warmup_steps(self):
        if self.debug:
            return

        rospy.sleep(1)
        for _ in trange(5, desc="Warmup Steps"):
            random_max_action = np.random.choice([0.999, -0.999], size=2*self.n_robots)
            self.env.step(random_max_action)


def main(args):
    rospy.init_node("collect_data")

    robot_pos_dict, robot_vel_dict, object_pos, object_vel, corner_pos, action_receipt_dict, tf_buffer, tf_listener = make_state_subscriber(args.robot_ids)
    data_collector = DataCollector(robot_pos_dict, robot_vel_dict, object_pos, object_vel, corner_pos, action_receipt_dict, **vars(args))
    data_collector.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-robot_ids', nargs='+', type=int, default=[0])
    parser.add_argument('-object_id', type=int, default=3)
    parser.add_argument("-n_samples", type=int, default=20)
    parser.add_argument("-buffer_capacity", type=int, default=10000)
    parser.add_argument('-exp_name', type=str, default=None)

    parser.add_argument("-use_object", action='store_true')
    parser.add_argument("-debug", action='store_true')
    parser.add_argument("-random_data", action='store_true')
    parser.add_argument("-beta", action='store_true')
    parser.add_argument("-policy", action='store_true')

    parser.add_argument('-cost_distance_weight', type=float, default=1.)
    parser.add_argument('-cost_distance_bonus_weight', type=float, default=0.)
    parser.add_argument('-cost_separation_weight', type=float, default=0.)
    parser.add_argument('-cost_std_weight', type=float, default=0.)
    parser.add_argument('-cost_goal_angle_weight', type=float, default=0.)
    parser.add_argument('-cost_realistic_weight', type=float, default=0.)
    parser.add_argument('-cost_action_weight', type=float, default=0.)
    parser.add_argument('-cost_to_object_heading_weight', type=float, default=0.)
    parser.add_argument('-cost_object_delta_weight', type=float, default=0.)

    args = parser.parse_args()

    if args.policy:
        args.n_robots = 2
        args.use_object = True
        args.robot_goals = True
        args.scale = True
        args.dist = True
        args.load_agent = False
        args.mpc_horizon = 3
        args.mpc_samples = 300
        args.gamma = 10.
        args.hidden_dim = 500
        args.hidden_depth = 3
        args.lr = 1e-3
        args.noise_std = 0.9
        args.beta = 0.8
        args.discount_factor = 0.99

    main(args)
