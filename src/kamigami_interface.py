#!/usr/bin/python3

import os
import pickle as pkl
from abc import abstractmethod

import numpy as np
import rospy

from replay_buffer import ReplayBuffer

from ros_stuff.srv import CommandAction
from ros_stuff.msg import RobotCmd

from ar_track_alvar_msgs.msg import AlvarMarkers
from tf.transformations import euler_from_quaternion


class KamigamiInterface:
    def __init__(self, robot_id, object_id, save_path, calibrate, new_buffer=False):
        self.save_path = save_path
        self.robot_id = robot_id
        self.object_id = object_id

        max_pwm = 0.999
        self.action_range = np.array([[-max_pwm, -max_pwm], [max_pwm, max_pwm]])
        self.duration = 0.3
        self.robot_state = np.zeros(3)      # (x, y, theta)
        self.object_state = np.zeros(3)     # (x, y, theta)

        self.n_updates = 0
        self.last_n_updates = 0
        self.not_found = False
        self.started = False

        self.n_avg_states = 1
        self.n_wait_updates = 1
        self.n_clip = 3
        self.flat_lim = 0.6
        self.save_freq = 50

        if os.path.exists("/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/replay_buffers/buffer.pkl") and not new_buffer:
            with open("/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/replay_buffers/buffer.pkl", "rb") as f:
                self.replay_buffer = pkl.load(f)
        else:
            self.replay_buffer = ReplayBuffer(capacity=10000, state_dim=6, action_dim=2, next_state_dim=6)

        self.states = []
        self.actions = []
        self.next_states = []
        self.done = False

        rospy.init_node("kamigami_interface")

        print(f"waiting for robot {self.robot_id} service")
        rospy.wait_for_service(f"/kami{self.robot_id}/server")
        self.service_proxy = rospy.ServiceProxy(f"/kami{self.robot_id}/server", CommandAction)
        print("connected to robot service")

        print("waiting for /ar_pose_marker rostopic")
        rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.update_state, queue_size=1)
        print("subscribed to /ar_pose_marker")

        self.tag_offset_path = "/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/sim/data/tag_offsets.npy"
        if not os.path.exists(self.tag_offset_path) or calibrate:
            self.calibrating = True
            self.calibrate()

        self.calibrating = False
        self.tag_offsets = np.load(self.tag_offset_path)

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def get_take_actions(self):
        pass

    def calibrate(self):
        if os.path.exists(self.tag_offset_path):
            tag_offsets = np.load(self.tag_offset_path)
        else:
            tag_offsets = np.zeros(10)

        input(f"Place robot/object on the left calibration point, aligned with the calibration line and hit enter.")
        left_state = self.get_state(wait=False)
        input(f"Place robot/object on the right calibration point, aligned with the calibration line and hit enter.")
        right_state = self.get_state(wait=False)

        robot_left_state, object_left_state = left_state[:3], left_state[3:]
        robot_right_state, object_right_state = right_state[:3], right_state[3:]

        true_robot_vector = (robot_left_state - robot_right_state)[:2]
        true_robot_angle = np.arctan2(true_robot_vector[1], true_robot_vector[0])

        true_object_vector = (object_left_state - object_right_state)[:2]
        true_object_angle = np.arctan2(true_object_vector[1], true_object_vector[0])

        measured_robot_angle = robot_left_state[2]
        measured_object_angle = object_left_state[2]

        tag_offsets[self.robot_id] = true_robot_angle - measured_robot_angle
        tag_offsets[self.object_id] = true_object_angle - measured_object_angle

        np.save(self.tag_offset_path, tag_offsets)

    def update_state(self, msg):
        found_robot, found_object = False, False
        for marker in msg.markers:
            if marker.id == self.robot_id:
                state = self.robot_state
                found_robot = True
            elif marker.id == self.object_id:
                state = self.object_state
                found_object = True
            else:
                continue

            o = marker.pose.pose.orientation
            o_list = [o.x, o.y, o.z, o.w]
            x, y, z = euler_from_quaternion(o_list)

            if abs(np.sin(x)) > self.flat_lim or abs(np.sin(y)) > self.flat_lim and self.started:
                print(f"{'ROBOT' if marker.id == self.robot_id else 'OBJECT'} MARKER NOT FLAT ENOUGH")
                print("sin(x):", np.sin(x), "|| sin(y)", np.sin(y))
                self.not_found = True
                return

            if hasattr(self, "tag_offsets"):
                z += self.tag_offsets[marker.id]

            state[0] = marker.pose.pose.position.x
            state[1] = marker.pose.pose.position.y
            state[2] = z % (2 * np.pi)

        self.not_found = not (found_robot and found_object)
        self.n_updates += 0 if self.not_found else 1

    def get_state(self, wait=True):
        if self.n_avg_states > 1 and not wait:
            robot_states, object_states = [], []
            while len(robot_states) < self.n_avg_states:
                if wait:
                    if self.n_updates == self.last_n_updates:
                        rospy.sleep(0.0001)
                    else:
                        robot_states.append(self.robot_state.copy())
                        object_states.append(self.object_state.copy())
                        self.last_n_updates = self.n_updates

            robot_state = np.array(robot_states).squeeze().mean(axis=0)
            object_state = np.array(object_states).squeeze().mean(axis=0)
        else:
            if wait:
                while self.n_updates == self.last_n_updates:
                    rospy.sleep(0.0001)
                self.last_n_updates = self.n_updates

            robot_state = self.robot_state.copy()
            object_state = self.object_state.copy()

        current_state = np.concatenate((robot_state, object_state), axis=0)
        return current_state

    def save_training_data(self, clip_end=False):
        states = np.array(self.states)
        actions = np.array(self.actions)
        next_states = np.array(self.next_states)

        length = min(len(states), len(actions), len(next_states))
        states = states[:length]
        actions = actions[:length]
        next_states = next_states[:length]

        if not os.path.exists(self.save_path):
            print("Creating new data!")
            np.savez_compressed(self.save_path, states=states, actions=actions, next_states=next_states)
        else:
            print("\nAppending new data to old data!")
            data = np.load(self.save_path)
            old_states = np.copy(data["states"])
            old_actions = np.copy(data["actions"])
            old_next_states = np.copy(data["next_states"])
            if all(len(old) != 0 for old in [old_states, old_actions, old_next_states]):
                states = np.append(old_states, states, axis=0)
                actions = np.append(old_actions, actions, axis=0)
                next_states = np.append(old_next_states, next_states, axis=0)

            # ignore last few transitions in case of uncertain ending
            if clip_end:
                states = states[:-self.n_clip]
                actions = actions[:-self.n_clip]
                next_states = next_states[:-self.n_clip]
                if len(states) + len(actions) + len(next_states) == 0:
                    print("skipping this save!")
                    return

            np.savez_compressed(self.save_path, states=states, actions=actions, next_states=next_states)

        print(f"Collected {len(states)} transitions in total!")
        self.states = []
        self.actions = []
        self.next_states = []
