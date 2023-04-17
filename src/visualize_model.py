#!/usr/bin/python

import argparse
import numpy as np
import matplotlib.pyplot as plt

import rospy
from ros_stuff.msg import MultiRobotCmd

from agents import MPPIAgent
from replay_buffer import ReplayBuffer
from train_utils import train_from_buffer
from utils import build_action_msg, make_state_subscriber, YAW_OFFSET_PATH


class Environment:
    def __init__(self, robot_pos_dict, robot_vel_dict, object_pos, object_vel, corner_pos, action_receipt_dict, params):
        self.params = params

        self.robot_pos_dict = robot_pos_dict
        self.robot_vel_dict = robot_vel_dict
        self.object_pos = object_pos
        self.object_vel = object_vel
        self.corner_pos = corner_pos
        self.action_receipt_dict = action_receipt_dict

        self.action_publisher = rospy.Publisher("/action_topic", MultiRobotCmd, queue_size=1)

        self.yaw_offsets = np.load(YAW_OFFSET_PATH)
        self.corner_pos_perm = corner_pos

        self.action_duration = 0.4
        self.post_action_sleep_time = 0.5

    def __getattr__(self, key):
        return self.params[key]

    def step(self, action):
        action_msg = build_action_msg(action, self.action_duration, self.robot_ids)
        self.action_publisher.publish(action_msg)

        while not all([receipt for receipt in self.action_receipt_dict.values()]):
            rospy.sleep(0.01)

        rospy.sleep(self.action_duration + self.post_action_sleep_time)
        for id in self.action_receipt_dict:
            self.action_receipt_dict[id] = False

        return self.get_state()

    def get_state(self):
        states = []

        for id in self.robot_pos_dict:
            states.append(self.get_state_from_id(id))

        states.append(self.get_state_from_id(self.object_id))

        out_of_bounds = lambda pos: np.any(pos[:2] > self.corner_pos[:2]) or np.any(pos[:2] < 0)

        valid = True
        for state in states:
            valid = valid and not out_of_bounds(state)

        if not valid:
            print("\nOUT OF BOUNDS\n")
            import pdb;pdb.set_trace()
            state = None
        else:
            state = np.concatenate(states, axis=0)

        return state

    def get_state_from_id(self, id):
        if id == self.object_id:
            pos = self.object_pos.copy()
        else:
            pos = self.robot_pos_dict[id].copy()

        pos[2] = (pos[2] + self.yaw_offsets[id]) % (2 * np.pi)
        return pos


def quiver_plot_states(states, color, labels=[None]*3):
    plt.quiver(states[:, 0], states[:, 1], np.cos(states[:, 2]), np.sin(states[:, 2]), color=color)

    markers = ['.', 'v', '*']
    for state, marker, label in zip(states, markers, label):
        plt.plot(state[0], state[1], marker=marker, color=color, label=label)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-robot_ids', nargs='+', type=int, default=[0])
    parser.add_argument('-object_id', type=int, default=3)

    parser.add_argument('-ensemble_size', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=10000)

    args = parser.parse_args()
    params = vars(args)

    robot_pos_dict, robot_vel_dict, object_pos, object_vel, corner_pos, action_receipt_dict, tf_buffer, tf_listener = make_state_subscriber(args.robot_ids)
    env = Environment(robot_pos_dict, robot_vel_dict, object_pos, object_vel, corner_pos, action_receipt_dict, params)

    replay_buffer = ReplayBuffer(params)
    replay_buffer.restore(restore_path='~/kamigami_data/replay_buffers/online_buffers/2object_tetherless.npz')
    print("\n\nREPLAY BUFFER SIZE:", replay_buffer.size, "\n")

    agent = MPPIAgent(params)
    train_from_buffer(
        agent, replay_buffer, validation_buffer=None,
        pretrain_samples=replay_buffer.size, save_agent=False,
        train_epochs=500, batch_size=1000, meta=False,
    )

    while True:
        action_str = input("\nProvide action to take: ")
        action = np.array([float(ac_str) for ac_str in action_str.split(', ')])

        state = env.get_state()
        pred_next_state = agent.simulate(state, action[None, None, :])
        next_state = env.step(action)

        plot_states = state.reshape(3, 3)
        plot_pred_next_states = pred_next_state.reshape(3, 3)
        plot_next_states = next_state.reshape(3, 3)

        quiver_plot_states(plot_states, 'black', labels=['robot 0', 'robot 2', 'object'])
        quiver_plot_states(plot_pred_next_states, 'blue')
        quiver_plot_states(plot_next_states, 'green')

        plt.legend()
        plt.show()
