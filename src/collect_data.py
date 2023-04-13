#!/usr/bin/python

import argparse
import numpy as np
import rospy
from tqdm import trange

from environment import Environment
from replay_buffer import ReplayBuffer
from utils import make_state_subscriber


class DataCollector:
    def __init__(self, robot_pos_dict, robot_vel_dict, object_pos, object_vel, corner_pos, action_receipt_dict, **params):
        self.params = params
        params["episode_length"] = np.inf
        params["robot_ids"].sort()
        params["n_robots"] = len(self.robot_ids)
        params["robot_goals"] = False
        params["buffer_save_dir"] = f"~/kamigami_data/replay_buffers/{'random_buffers' if self.random_data else 'meshgrid_buffers'}/"

        # states
        self.env = Environment(robot_pos_dict, robot_vel_dict, object_pos, object_vel, corner_pos, action_receipt_dict, params, precollecting=True)
        self.replay_buffer = ReplayBuffer(params, precollecting=True)

        action_range = np.linspace(-1, 1, np.floor(np.sqrt(self.n_samples)).astype("int"))
        left_actions, right_actions = np.meshgrid(action_range, action_range)
        self.fixed_actions = np.stack((left_actions, right_actions)).transpose(2, 1, 0).reshape(-1, 2)

    def __getattr__(self, key):
        return self.params[key]

    def run(self):
        self.take_warmup_steps()

        state = self.env.get_state()

        print("\nCOLLECTING DATA\n")

        for i in trange(self.n_samples):
            valid = False

            while not valid:
                if self.random_data:
                    if self.beta:
                        beta_param = 0.7
                        action = np.random.beta(beta_param, beta_param, size=2*self.n_robots) * 2 - 1
                    else:
                        action = np.random.uniform(low=-1., high=1., size=2*self.n_robots)
                else:
                    action = self.fixed_actions[i]

                next_state, _ = self.env.step(action)

                if state is not None and next_state is not None:
                    self.replay_buffer.add(state, action, next_state)
                    state = next_state
                    valid = True
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

    args = parser.parse_args()
    main(args)
