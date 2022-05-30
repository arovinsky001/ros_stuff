#!/usr/bin/python3

import os
import numpy as np
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
        action = 0.9
        actions_discrete = np.array([-0.6, -0.55, -0.5, -0.45, 0.45, 0.5, 0.55, 0.6])
        xa, ya = np.meshgrid(actions_discrete, actions_discrete)
        # self.actions_table = np.stack((xa, ya)).transpose(1, 2, 0).reshape(-1, 2)
        self.actions_table = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]) * 0.95

        self.min_action = -action
        self.max_action = action
        self.current_state = np.zeros(3)        # (x, y, theta)
        # self.base_state = np.zeros(3)
        self.robot_id = 2
        # self.base_id = 1
        self.states = []
        self.actions = []
        self.next_states = []
        self.step_count = 0
        self.found_count = 0
        self.n_updates = 0
        rospy.init_node("data_collector")
        self.time = rospy.get_rostime().to_sec()
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
            # found_base = False
            for marker in msg.markers:
                if marker.id == self.robot_id:
                    state = self.current_state
                    found_robot = True
                # elif marker.id == self.base_id:
                #     state = self.base_state
                #     found_base = True
                else:
                    continue
                
                state[0] = marker.pose.pose.position.x
                state[1] = marker.pose.pose.position.y
                
                o = marker.pose.pose.orientation
                o_list = [o.x, o.y, o.z, o.w]
                _, _, z = euler_from_quaternion(o_list)
                state[2] = z % (2 * np.pi)
            
            # if not found_robot and found_base:
            if not found_robot:
                self.found_count += 1
            else:
                self.found_count = 0
        except:
            print("could not update states")
            import pdb;pdb.set_trace()
        self.n_updates += 1
        
    def collect_data(self):
        if self.found_count >= 20:
            print("\n\nRobot hit boundary!\n\n")
            self.save_data(clip_end=True)
            rospy.signal_shutdown("bye")
            return
        
        if self.found_count > 0:
            action_req = RobotCmd()
            action_req.left_pwm = 0.9
            action_req.right_pwm = -0.9
            self.command_action(action_req, 'kami1')
            print("PERTURBING")
            rospy.sleep(0.05)
            return
        
        if self.n_updates == 0:
            return

        if self.step_count % len(self.actions_table) == 0:
            np.random.shuffle(self.actions_table)
        
        # current_state = self.current_state - self.base_state
        n_updates = self.n_updates
        current_states = []
        while len(current_states) < 50:
            if self.n_updates == n_updates:
                continue
            n_updates = self.n_updates
            current_states.append(self.current_state)
        
        current_states = np.array(current_states)
        import pdb;pdb.set_trace()
        current_state = current_states.mean(axis=0).squeeze()
        theta = current_state[2]
        state = np.array([current_state[0], current_state[1], np.sin(theta), np.cos(theta)])

        # action = np.random.uniform(low=self.min_action, high=self.max_action, size=2)
        # min_magnitude = 0.4
        # if not np.any(np.abs(action) > min_magnitude):
        #     action[action.argmax()] = min_magnitude * np.sign(action[action.argmax()])
        
        action = self.actions_table[self.step_count % len(self.actions_table)]

        # if rospy.get_rostime().to_sec() - self.time > 1.0:
        #     self.time = rospy.get_rostime().to_sec()
        print(f"\nstate:, {current_state}")
        print(f"action: {action}")

        action_req = RobotCmd()
        action_req.left_pwm = action[0]
        action_req.right_pwm = action[1]

        self.command_action(action_req, 'kami1')      
        rospy.sleep(0.1)
        n_updates = self.n_updates
        
        while self.n_updates - n_updates < 5:
            rospy.sleep(0.001)

        if self.found_count > 0:
            return
        
        n_updates = self.n_updates
        next_states = []
        while len(next_states) < 5:
            if self.n_updates == n_updates:
                continue
            n_updates = self.n_updates
            next_states.append(self.current_state)
        
        next_state = np.array(next_states).mean(axis=0).squeeze()
        theta = next_state[2]
        next_state = np.array([next_state[0], next_state[1], np.sin(theta), np.cos(theta)])

        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.step_count += 1
        
        if self.step_count % 10 == 0:
            self.save_data()
    
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
    
