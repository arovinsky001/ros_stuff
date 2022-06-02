#!/usr/bin/env python3

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

import rospy
from ar_track_alvar_msgs.msg import AlvarMarkers
from ros_stuff.srv import CommandAction
from ros_stuff.msg import RobotCmd
from tf.transformations import euler_from_quaternion
from forward_mpc_agent import *



class RealMPC:
    def __init__(self, kami_ids, agent_path, goal, mpc_steps, mpc_samples):
        self.n = 0
        self.n_updates = 0
        self.n_avg_states = 2
        self.n_wait_updates = 3
        self.prev_actions = np.zeros((len(kami_ids), 2, 2))
        self.online = False

        self.kami_ids = np.array(kami_ids)
        self.agents = []
        self.goal = goal
        self.tol = 0.07
        self.dones = np.array([False] * len(kami_ids))
        self.states = np.zeros((len(kami_ids), 3))
        self.state_range = np.array([-np.inf, np.inf])
        action = 0.9
        self.action_range = np.array([-action, action])
        self.mpc_steps = mpc_steps
        self.mpc_samples = mpc_samples
        self.dists = []
        for _ in self.kami_ids:
            with open(agent_path, "rb") as f:
                self.agents.append(pkl.load(f))
                self.agents[-1].model.eval()
        rospy.init_node('laptop_client')

        # Get info on positioning from camera & AR tags
        print("waiting for service")
        rospy.wait_for_service('/kami1/server')
        self.command_action = rospy.ServiceProxy('/kami1/server', CommandAction)
        rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.update_state, queue_size=1)

        while not rospy.is_shutdown():
            self.run()

    def update_state(self, msg):
        try:
            found_robot = False
            for marker in msg.markers:
                if marker.id in self.kami_ids:
                    idx = np.argwhere(self.kami_ids == marker.id)
                    if len(self.dones == 1):
                        if self.dones[0]:
                            continue
                        state = self.states[0]
                    else:
                        if self.dones[idx]:
                            continue
                        state = self.states[idx]
                    found_robot = True
                else:
                    continue

                o = marker.pose.pose.orientation
                o_list = [o.x, o.y, o.z, o.w]
                x, y, z = euler_from_quaternion(o_list)

                if abs(np.sin(x)) > 0.6 or abs(np.cos(x)) > 0.6:
                    return

                state[0] = marker.pose.pose.position.x
                state[1] = marker.pose.pose.position.y
                state[2] = z % (2 * np.pi)

            if found_robot:
                self.n_updates += 1
        except:
            print(f"could not update states, id: {marker.id}")
            import pdb;pdb.set_trace()
        

    def run(self):
        if self.n_updates == 0:
            return

        n_updates = self.n_updates
        states = self.get_states()

        if not hasattr(self, "init_states"):
            self.init_states = states
        diff = self.goal - states
        distances = np.linalg.norm(diff[:, :2], axis=-1).squeeze()
        print("DISTANCES:", distances)
        self.dists.append(distances)
        self.dones[distances < self.tol] = True
        
        if np.all(self.dones) or self.n == 100:
            plt.plot(np.arange(len(self.dists)), self.dists, "b-")
            plt.xlabel("Step")
            plt.ylabel("Distance to Goal")
            plt.title("Kamigami MPC")
            plt.show()
            rospy.signal_shutdown("Finished! All robots reached goal.")

        for i, agent in enumerate(self.agents):
            if not self.dones[i]:
                action = agent.mpc_action(states[i], self.init_states[i], self.goal, self.prev_actions[i],
                                        self.state_range, self.action_range, swarm=False, n_steps=self.mpc_steps,
                                        n_samples=self.mpc_samples, swarm_weight=0.0, perp_weight=0.4,
                                        heading_weight=0.17, forward_weight=0.0, dist_weight=1.0, norm_weight=0.1).detach().numpy()

                action_req = RobotCmd()
                action_req.left_pwm = action[0]
                action_req.right_pwm = action[1]
                print("\nACTION:", action)
                print("STATE:", states[i], '\n')
                self.command_action(action_req, f"kami{self.kami_ids[i]}")
                self.prev_actions[i, 0] = self.prev_actions[i, 1]
                self.prev_actions[i, 1] = action

        n_updates = self.n_updates
        while self.n_updates - n_updates < self.n_wait_updates:
            rospy.sleep(0.01)

        if self.online:
            next_states = []
            while len(next_states) < self.n_avg_states:
                if self.n_updates == n_updates:
                    continue
                n_updates = self.n_updates
                next_states.append(self.states[0].copy())
            
            next_states = np.array(next_states).mean(axis=0).squeeze()[None, :]
            theta = next_states[:, -1]
            next_states = np.append(np.append(next_states[:, :-1], np.sin(theta)[:, None], axis=1), np.cos(theta)[:, None], axis=1)
            
            states = (states - self.agents[0].states_mean) / self.agents[0].states_std
            action = (action - self.agents[0].actions_mean) / self.agents[0].actions_std
            next_states = (next_states - self.agents[0].next_states_mean) / self.agents[0].next_states_std
            self.agents[0].model.update(states, action[None, :], next_states)
        
        self.n += 1
    
    def get_states(self):
        current_states = []
        n_updates = self.n_updates
        while len(current_states) < self.n_avg_states:
            if self.n_updates == n_updates:
                continue
            n_updates = self.n_updates
            current_states.append(self.states.copy())
        
        current_states = np.array(current_states).squeeze()
        if len(current_states.shape) == 2:
            current_states = current_states.mean(axis=0, keepdims=True)
        else:
            current_states = current_states.mean(axis=1)
        theta = current_states[:, -1]
        states = np.block([current_states[:, :-1], np.sin(theta)[:, None], np.cos(theta)[:, None]])
        return states


if __name__ == '__main__':
    kami_ids = [0]
    agent_path = "/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/agents/real.pkl"

    goal = np.array([-1.0,  -0.9, 0.0, 1.0])
    mpc_steps = 2
    mpc_samples = 1000
    r = RealMPC(kami_ids, agent_path, goal, mpc_steps, mpc_samples)
