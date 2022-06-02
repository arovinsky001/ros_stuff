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
RAW_SAVE_PATH = "/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/sim/data/real_data_raw.npz"
SAVE_PATH = "/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/sim/data/real_data.npz"

class DataCollector:
    def __init__(self):
        self.prev_actions = []
        self.n_avg_states = 2
        self.n_wait_updates = 3
        self.action_range = [-1.0, 1.0]
        self.current_state = np.zeros(3)        # (x, y, theta)
        self.robot_id = 0
        self.states = []
        self.actions = []
        self.next_states = []
        self.step_count = 0
        self.found_count = 0
        self.n_updates = 0
        self.perturb_count = 0
        rospy.init_node("data_collector")
        print("waiting for service")
        rospy.wait_for_service("/kami1/server")
        self.command_action = rospy.ServiceProxy("/kami1/server", CommandAction)
        rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.update_state, queue_size=1)
        # rospy.spin()
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

                if abs(np.sin(x)) > 0.6 or abs(np.cos(x)) > 0.6:
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

        if not self.check_found_perturb():
            return

        state = self.get_state()
        action = self.get_command_action()

        if not self.check_found_perturb():
            return

        next_state = self.get_state()

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
        theta = current_state[2]
        state = np.array([current_state[0], current_state[1], np.sin(theta), np.cos(theta)])
        return state

    def check_found_perturb(self):
        if self.perturb_count >= 5:
            print("\n\nRobot hit boundary!\n\n")
            self.save_data(clip_end=True)
            rospy.signal_shutdown("bye")
            return False

        if self.found_count > 0:
            action_req = RobotCmd()
            action_req.left_pwm = 0.9
            action_req.right_pwm = -0.9
            self.command_action(action_req, 'kami1')
            print("PERTURBING")
            rospy.sleep(0.05)
            self.perturb_count += 1
            return False

        self.perturb_count = 0
        return True
    
    def get_command_action(self):
        action = np.random.uniform(*action_range, size=2)
        action_req = RobotCmd()
        action_req.left_pwm = action[0]
        action_req.right_pwm = action[1]
        self.command_action(action_req, 'kami1')      
        n_updates = self.n_updates
        while self.n_updates - n_updates < self.n_wait_updates:
            rospy.sleep(0.01)
        return action
    
    def save_data(self, clip_end=False):
        states = np.array(self.states)
        actions = np.array(self.actions)
        next_states = np.array(self.next_states)

        if clip_end:
            clip = 3
            states = states[:-3]
            actions = actions[:-3]
            next_states = next_states[:-3]
            if len(states) == 0:
                print("No new states to append!")
                return
        
        length = min(len(states), len(actions), len(next_states))
        states = states[:length]
        actions = actions[:length]
        next_states = next_states[:length]

        if not os.path.exists(RAW_SAVE_PATH):
            print("Creating new data!")
            np.savez_compressed(RAW_SAVE_PATH, states=states, actions=actions, next_states=next_states)
        else:
            try:
                print("\nAppending new data to old data!")
                data = np.load(RAW_SAVE_PATH)
                old_states = np.copy(data["states"])
                old_actions = np.copy(data["actions"])
                old_next_states = np.copy(data["next_states"])
                import pdb;pdb.set_trace()
                if len(old_states) != 0 and len(old_actions) != 0:
                    states = np.append(old_states, states, axis=0)
                    actions = np.append(old_actions, actions, axis=0)
                    next_states = np.append(old_next_states, next_states, axis=0)
                np.savez_compressed(RAW_SAVE_PATH, states=states, actions=actions, next_states=next_states)
            except:
                import pdb;pdb.set_trace()
        self.states = []
        self.actions = []
        self.next_states = []
        print(f"Collected {len(states)} transitions in total!")

def load_and_process():
    data = np.load(RAW_SAVE_PATH)
    states = np.array(data['states'])
    actions = np.array(data['actions'])
    next_states = np.array(data['next_states'])
    # states = states[:-30]
    # actions = actions[:-30]
    # next_states = next_states[:-30]
    # np.savez_compressed(RAW_SAVE_PATH, states=states, actions=actions, next_states=next_states)
    length = min(len(states), len(actions), len(next_states))
    states = states[:length]
    actions = actions[:length]
    next_states = next_states[:length]
    
    np.savez_compressed(SAVE_PATH, states=states, actions=actions, next_states=next_states)

if __name__ == '__main__':
    dc = DataCollector()
    # load_and_process()
    
    # try:
    #     dc.process_raw_data()
    # except:
    #     print("could not process raw data")
    #     import pdb;pdb.set_trace()
    
