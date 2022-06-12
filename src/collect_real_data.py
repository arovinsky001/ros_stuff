#!/usr/bin/python3

import os
import numpy as np
from matplotlib import animation

from ar_track_alvar_msgs.msg import AlvarMarkers
from ros_stuff.srv import CommandAction
from ros_stuff.msg import RobotCmd
from tf.transformations import euler_from_quaternion

import rospy

AVG_STEPS = 2
SAVE_PATH = "/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/sim/data/real_data.npz"

class DataCollector:
    def __init__(self):
        self.prev_actions = []
        self.n_avg_states = 2
        self.n_wait_updates = 2
        self.max_perturb_count = 5
        self.n_clip = 3
        self.action_range = np.array([[-0.99, -0.99, 0.05], [0.99, 0.99, 0.6]])
        self.current_state = np.zeros(3)        # (x, y, theta)
        self.robot_id = 0
        self.states = []
        self.actions = []
        self.next_states = []
        self.step_count = 0
        self.found_count = 0
        self.n_updates = 0
        self.perturb_count = 0
        self.flat_lim = 0.6
        rospy.init_node("data_collector")
        print("waiting for service")
        rospy.wait_for_service("/kami1/server")
        self.command_action = rospy.ServiceProxy("/kami1/server", CommandAction)
        rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.update_state, queue_size=1)
        while not rospy.is_shutdown():
            self.collect_data()

    def update_state(self, msg):
        try:
            found_robot = False
            for marker in msg.markers:
                if marker.id == self.robot_id:
                    state = self.current_state
                    found_robot = True
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
            
            if not found_robot:
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

        if len(self.prev_actions) == 0:
            self.prev_actions.append(self.get_command_action())
            self.prev_actions.append(self.get_command_action())
            return

        state = self.get_state()
        if state is None:
            return

        action = self.get_command_action()

        next_state = self.get_state()
        if next_state is None:
            return

        print(f"\nstate:, {self.current_state}")
        print(f"action: {action}")

        actions3 = np.block([self.prev_actions[0], self.prev_actions[1], action])
        self.prev_actions[0] = self.prev_actions[1]
        self.prev_actions[1] = action
        
        self.states.append(state)
        self.actions.append(actions3)
        self.next_states.append(next_state)
        self.step_count += 1
        
        if self.step_count % 5 == 0:
            self.save_data()
    
    def get_state(self):
        if self.perturb_count >= self.max_perturb_count:
            print("\n\nRobot hit boundary!\n\n")
            self.save_data(clip_end=True)
            rospy.signal_shutdown("bye")
            return None

        if self.found_count > 0:
            action_req = RobotCmd()
            action_req.left_pwm = 0.9
            action_req.right_pwm = -0.9
            self.command_action(action_req, 'kami1')
            print("PERTURBING")
            rospy.sleep(0.01)
            self.perturb_count += 1
            return None
        
        self.perturb_count = 0

        current_states = []
        n_updates = self.n_updates
        while len(current_states) < self.n_avg_states:
            if self.n_updates == n_updates:
                continue
                print("continuing")
            n_updates = self.n_updates
            current_states.append(self.current_state.copy())
        
        current_states = np.array(current_states)
        current_state = current_states.mean(axis=0).squeeze()

        return current_state
    
    def get_command_action(self):
        action = np.random.uniform(*self.action_range, size=self.action_range.shape[-1])
        action_req = RobotCmd()
        action_req.left_pwm = action[0]
        action_req.right_pwm = action[1]
        action_req.duration = action[2]
        self.command_action(action_req, 'kami1')      
        n_updates = self.n_updates
        i = 0
        while self.n_updates - n_updates < self.n_wait_updates:
            if i == 100:
                import pdb;pdb.set_trace()
            rospy.sleep(0.01)
            i += 1
        return action
    
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
