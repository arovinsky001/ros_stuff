#!/usr/bin/python3

import os
import numpy as np

from ar_track_alvar_msgs.msg import AlvarMarkers
from ros_stuff.srv import CommandAction
from ros_stuff.msg import RobotCmd
from tf.transformations import euler_from_quaternion

import rospy

SAVE_PATH = "/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/sim/data/real_data.npz"

class DataCollector:
    def __init__(self):
        self.action_range = np.array([[-0.99, -0.99, 0.1], [0.99, 0.99, 0.6]])
        self.robot_ids = np.array([0, 2])

        self.n_avg_states = 1
        self.n_wait_updates = 1
        self.max_perturb_count = 5
        self.n_clip = 3
        self.step_count = 0
        self.found_count = 0
        self.n_updates = 0
        self.perturb_count = 0
        self.flat_lim = 0.6
        
        self.current_states = np.zeros((len(self.robot_ids), 4))        # (x, y, theta)
        self.states = []
        self.actions = []
        self.next_states = []
        
        rospy.init_node("data_collector")
        print("waiting for service")
        self.service_proxies = []
        for id in self.robot_ids:
            rospy.wait_for_service(f"/kami{id}/server")
            self.service_proxies.append(rospy.ServiceProxy(f"/kami{id}/server", CommandAction))
        rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.update_state, queue_size=1)
        while not rospy.is_shutdown():
            self.collect_data()

    def update_state(self, msg):
        try:
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
                    print(np.sin(x), np.sin(y))
                    self.found_count += 1
                    return

                state[0] = marker.pose.pose.position.x
                state[1] = marker.pose.pose.position.y
                state[2] = z % (2 * np.pi)
                state[3] = marker.id
            
            if not np.all(found_robots):
                self.found_count += 1
            else:
                self.found_count = 0
                self.n_updates += 1
        except:
            print("could not update states")
            import pdb;pdb.set_trace()
        
    def collect_data(self):
        if self.n_updates == 0:
            return

        states = self.get_states()
        if states is None:
            return

        actions = self.get_command_actions()

        next_states = self.get_states()
        if next_states is None:
            return

        print(f"\nstates:, {self.current_states}")
        print(f"actions: {actions}")
        
        self.states.append(states)
        self.actions.append(actions)
        self.next_states.append(next_states)
        self.step_count += 1
        
        if self.step_count % 10 == 0:
            self.save_data()
    
    def get_states(self):
        if self.perturb_count >= self.max_perturb_count:
            print("\n\nRobot hit boundary!\n\n")
            self.save_data(clip_end=True)
            rospy.signal_shutdown("bye")
            return None

        if self.found_count > 0:
            print("PERTURBING")

            reqs = [RobotCmd() for _ in range(len(self.robot_ids))]
            for req in reqs:
                req.left_pwm = 0.9
                req.right_pwm = -0.9
                req.duration = 0.2
            
            for i, proxy in enumerate(self.service_proxies):
                proxy(reqs[i], f'kami{i+1}')

            rospy.sleep(0.01)
            self.perturb_count += 1
            return None
        
        self.perturb_count = 0

        if self.n_avg_states > 1:
            current_states = []
            n_updates = self.n_updates
            while len(current_states) < self.n_avg_states:
                if self.n_updates == n_updates:
                    continue
                n_updates = self.n_updates
                current_states.append(self.current_states.copy())
            
            current_states = np.array(current_states).squeeze()
            
            keepdims = (len(current_states.shape) == 2)
            current_states = current_states.mean(axis=0, keepdims=keepdims)

            return current_states
        else:
            return self.current_states.copy().squeeze()


        
    
    def get_command_actions(self):
        actions = np.random.uniform(*self.action_range, size=(len(self.robot_ids), self.action_range.shape[-1]))
        actions = np.append(actions, np.empty((len(self.robot_ids), 1)), axis=1)
        reqs = [RobotCmd() for _ in range(len(self.robot_ids))]
        for i, req in enumerate(reqs):
            req.left_pwm = actions[i, 0]
            req.right_pwm = actions[i, 1]
            req.duration = actions[i, 2]
            actions[i, 3] = self.robot_ids[i]
        
        for i, proxy in enumerate(self.service_proxies):
            proxy(reqs[i], f'kami{self.robot_ids[i]}')

        n_updates = self.n_updates
        i = 0
        while self.n_updates - n_updates < self.n_wait_updates:
            if i == 200:
                print("DIDN'T UPDATE FOR TOO LONG")
                import pdb;pdb.set_trace()
            rospy.sleep(0.01)
            i += 1

        return actions
    
    def save_data(self, clip_end=False):
        states = np.array(self.states)
        actions = np.array(self.actions)
        next_states = np.array(self.next_states)

        if clip_end:
            states = states[:-self.n_clip]
            actions = actions[:-self.n_clip]
            next_states = next_states[:-self.n_clip]
            if len(states) == 0:
                print("No new states to append!")
                return
        
        length = min(len(states), len(actions), len(next_states))
        states = states[:length]
        actions = actions[:length]
        next_states = next_states[:length]

        if not os.path.exists(SAVE_PATH):
            print("Creating new data!")
            np.savez_compressed(SAVE_PATH, states=states, actions=actions, next_states=next_states)
        else:
            try:
                print("\nAppending new data to old data!")
                data = np.load(SAVE_PATH)
                old_states = np.copy(data["states"])
                old_actions = np.copy(data["actions"])
                old_next_states = np.copy(data["next_states"])
                if len(old_states) != 0 and len(old_actions) != 0:
                    states = np.append(old_states, states, axis=0)
                    actions = np.append(old_actions, actions, axis=0)
                    next_states = np.append(old_next_states, next_states, axis=0)
                np.savez_compressed(SAVE_PATH, states=states, actions=actions, next_states=next_states)
            except:
                import pdb;pdb.set_trace()
        self.states = []
        self.actions = []
        self.next_states = []
        print(f"Collected {len(states)} transitions in total!")

if __name__ == '__main__':
    dc = DataCollector()
