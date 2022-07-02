#!/usr/bin/python3

import os
from abc import abstractmethod

import numpy as np
import rospy

from ar_track_alvar_msgs.msg import AlvarMarkers
from ros_stuff.srv import CommandAction
from ros_stuff.msg import RobotCmd
from tf.transformations import euler_from_quaternion


class KamigamiInterface:
    def __init__(self, robot_ids, save_path):
        self.save_path = save_path
        self.robot_ids = np.array(robot_ids)

        max_pwm = 0.999
        self.action_range = np.array([[-max_pwm, -max_pwm, 0.1], [max_pwm, max_pwm, 0.6]])
        self.current_states = np.zeros((len(self.robot_ids), 4))    # (x, y, theta, id)

        # for perturbing robots in case not visible
        self.perturb_request = RobotCmd()
        self.perturb_request.left_pwm = 0.9
        self.perturb_request.right_pwm = -0.9
        self.perturb_request.duration = 0.2

        self.perturb_count = 0
        self.n_updates = 0
        self.not_found = False
        
        self.n_avg_states = 4
        self.n_wait_updates = 4
        self.max_perturb_count = 5
        self.n_clip = 3
        self.flat_lim = 0.6
        self.save_freq = 10

        self.states = []
        self.actions = []
        self.next_states = []
        self.done = False

        rospy.init_node("kamigami_interface")

        self.service_proxies = []
        for id in self.robot_ids:
            print(f"waiting for robot {id} service")
            rospy.wait_for_service(f"/kami{id}/server")
            self.service_proxies.append(rospy.ServiceProxy(f"/kami{id}/server", CommandAction))
        print("connected to all robot services")

        print("waiting for /ar_pose_marker rostopic")
        rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.update_states, queue_size=1)
        print("subscribed to /ar_pose_marker")

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def get_take_actions(self):
        pass

    def update_states(self, msg):
        found_robots = [False] * len(self.robot_ids)
        for marker in msg.markers:
            if marker.id in self.robot_ids:
                idx = np.argwhere(self.robot_ids == marker.id).squeeze().item()
                state = self.current_states[idx]
                found_robots[idx] = True
            else:
                continue
                            
            o = marker.pose.pose.orientation
            o_list = [o.x, o.y, o.z, o.w]
            x, y, z = euler_from_quaternion(o_list)

            if abs(np.sin(x)) > self.flat_lim or abs(np.sin(y)) > self.flat_lim:
                print("MARKER NOT FLAT ENOUGH")
                print("sin(x):", np.sin(x), "|| sin(y)", np.sin(y))
                self.not_found = True
                return

            state[0] = marker.pose.pose.position.x
            state[1] = marker.pose.pose.position.y
            state[2] = z % (2 * np.pi)
            state[3] = marker.id
        
        self.not_found = not np.all(found_robots)
        self.n_updates += 0 if self.not_found else 1

    def get_states(self, perturb=False):
        if perturb:
            if self.perturb_count >= self.max_perturb_count:
                print("\n\nRobot hit boundary!\n\n")
                self.save_data(clip_end=True)
                self.done = True
                return

            if self.not_found:
                print("PERTURBING")
                
                for i, proxy in enumerate(self.service_proxies):
                    proxy(self.perturb_request, f'kami{i+1}')

                rospy.sleep(0.1)
                self.perturb_count += 1
                return
            
        self.perturb_count = 0

        if self.n_avg_states > 1:
            current_states = []
            n_updates = self.n_updates
            while len(current_states) < self.n_avg_states:
                if self.n_updates == n_updates:
                    rospy.sleep(0.001)
                n_updates = self.n_updates
                current_states.append(self.current_states.copy())
            
            current_states = np.array(current_states).squeeze()
            
            keepdims = (len(current_states.shape) == 2)
            current_states = current_states.mean(axis=0, keepdims=keepdims)

            return current_states
        else:
            return self.current_states.copy().squeeze()

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
