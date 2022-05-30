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
        self.kami_ids = np.array(kami_ids)
        self.base_id = 1
        self.base_state = np.zeros(3)
        self.agents = []
        self.goal = goal
        self.tol = 0.07
        self.dones = np.array([False] * len(kami_ids))
        self.states = np.zeros((len(kami_ids), 3))
        self.state_range = np.array([[-np.inf], [np.inf]])
        action = 0.6
        self.action_range = np.array([[-action], [action]])
        self.mpc_steps = mpc_steps
        self.mpc_samples = mpc_samples
        self.dists = []
        for _ in self.kami_ids:
            with open(agent_path, "rb") as f:
                self.agents.append(pkl.load(f))
        rospy.init_node('laptop_client')

        # Get info on positioning from camera & AR tags
        rospy.wait_for_service('/kami1/server')
        self.command_action = rospy.ServiceProxy('/kami1/server', CommandAction)
        rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.callback, queue_size=1)

        rospy.spin()

    def callback(self, msg):
        try:
            for marker in msg.markers:
                if marker.id == self.base_id:
                    state = self.base_state
                elif marker.id in self.kami_ids:
                    if len(self.dones == 1):
                        if self.dones[0]:
                            continue
                        state = self.states[0]
                    else:
                        if self.dones[self.kami_ids == marker.id]:
                            continue
                        state = self.states[self.kami_ids == marker.id]
                else:
                    print("\nSHOULD NOT HAVE BEEN REACHED\n")
                    raise ValueError
                
                state[0] = marker.pose.pose.position.x
                state[1] = marker.pose.pose.position.y

                o = marker.pose.pose.orientation
                o_list = [o.x, o.y, o.z, o.w]
                _, _, z = euler_from_quaternion(o_list)
                state[2] = z % (2 * np.pi)
        except:
            print(f"could not update states, id: {marker.id}")
            import pdb;pdb.set_trace()

        states = self.states - self.base_state
        states[:, 2] %= 2 * np.pi
        # print(states)
        diff = self.goal - states
        distances = np.linalg.norm(diff[:, :-1], axis=-1)
        # tdist1 = diff[:, -1] % (2*np.pi)
        # tdist2 = 2 * np.pi - (diff[:, -1] % (2 * np.pi))
        # dists_theta = np.stack((tdist1, tdist2)).min(axis=0)
        # distances = dists_xy + 0.00 * dists_theta
        self.dists.append(distances[0])
        print("DISTANCE:", distances)
        self.dones[distances < self.tol] = True
        
        if np.all(self.dones):
            rospy.signal_shutdown("Finished! All robots reached goal.")

        for i, agent in enumerate(self.agents):
            if not self.dones[i]:
                action = agent.mpc_action(states[i], self.goal, self.state_range,
                                        self.action_range, n_steps=self.mpc_steps,
                                        n_samples=self.mpc_samples, swarm=False, swarm_weight=0.3).detach().numpy()

                action_req = RobotCmd()
                # lim = 0.4
                # if action[0] > 0.1:
                #     action[0] = np.clip(action[0], lim, np.inf)
                # elif action[0] < -0.1:
                #     action[0] = np.clip(action[0], -np.inf, -lim)
                # if action[1] > 0.1:
                #     action[1] = np.clip(action[1], lim, np.inf)
                # elif action[1] < -0.1:
                #     action[1] = np.clip(action[1], -np.inf, -lim)
                action_req.left_pwm = action[0]
                action_req.right_pwm = action[1]
                print("ACTION:", action)
                print("STATE:", states[i], '\n')
                self.command_action(action_req, f"kami{self.kami_ids[i]}")
        rospy.sleep(0.3)


if __name__ == '__main__':
    kami_ids = [0]
    agent_path = "/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/agents/real.pkl"

    goal = np.array([-0.6,  0.2, 0.0])
    # o_list = [-0.6463786551245982, 0.7234517765775437, -0.22730671518678613, -0.08452111213835743]
    # _, _, z = euler_from_quaternion(o_list)
    # goal = np.array([-0.15974444619689435, -0.5624425100428407, z])

    # o_list = [-0.6480803396139354, 0.7220224106196633, -0.22645192060868, -0.08599441622277922]
    # _, _, z = euler_from_quaternion(o_list)
    # goal = np.array([-0.1623737546700545, -0.560986043477003, z])
    mpc_steps = 5
    mpc_samples = 3000
    r = RealMPC(kami_ids, agent_path, goal, mpc_steps, mpc_samples)

    plt.plot(np.arange(len(r.dists)), r.dists, 'b-')