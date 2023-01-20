#!/usr/bin/python

import argparse
import numpy as np
import rospy
from tqdm import trange

from ros_stuff.msg import ProcessedStates, RobotCmd
from std_msgs.msg import Time
import tf2_ros

from environment import Environment
from replay_buffer import ReplayBuffer
from utils import make_state_subscriber


class DataCollector:
    def __init__(self, robot_pos, object_pos, corner_pos, robot_vel, object_vel, action_timestamp, params):
        self.params = params
        params["episode_length"] = np.inf

        # states
        self.robot_vel = robot_vel
        self.object_vel = object_vel
        self.state_dict = {
            "robot": robot_pos,
            "object": object_pos,
            "corner": corner_pos,
        }
        self.action_timestamp = action_timestamp
        self.last_action_timestamp = self.action_timestamp.copy()

        self.env = Environment(robot_pos, object_pos, corner_pos, robot_vel, object_vel, action_timestamp, params, None)
        self.replay_buffer = ReplayBuffer(params)

        action_range = np.linspace(-1, 1, np.floor(np.sqrt(self.n_samples)))
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
                    action = np.random.uniform(low=-1., high=1., size=2)
                else:
                    action = self.fixed_actions[i]

                next_state, _ = self.env.step(action)

                if state and next_state:
                    self.replay_buffer.add(state, action, next_state)
                    state = next_state
                    valid = True
                else:
                    state = self.env.get_state()

        print("\nDATA COLLECTED, SAVING REPLAY BUFFER\n")
        self.replay_buffer.dump()

    def take_warmup_steps(self):
        if self.debug:
            return

        for _ in trange(5, desc="Warmup Steps"):
            random_max_action = np.random.choice([0.999, -0.999], size=2)
            self.env.step(random_max_action)


def main(args):
    rospy.init_node("collect_data")

    robot_pos, object_pos, corner_pos, robot_vel, object_vel, action_timestamp, tf_buffer, tf_listener = make_state_subscriber()


    """
    run experiment
    """
    data_collector = DataCollector(robot_pos, object_pos, corner_pos, robot_vel, object_vel, action_timestamp, **vars(args))
    data_collector.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-debug", default=False)
    parser.add_argument("-n_samples", default=20)
    parser.add_argument("-random_data", default=False)

    args = parser.parse_args()
    main(args)
