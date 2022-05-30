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
        self.kami_ids = np.array(kami_ids)
        # self.base_id = 1
        # self.base_state = np.zeros(3)
        self.agents = []
        self.goal = goal
        self.tol = 0.07
        self.dones = np.array([False] * len(kami_ids))
        self.states = np.zeros((len(kami_ids), 3))
        self.state_range = np.array([-np.inf, np.inf])
        action = 0.85
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
            for marker in msg.markers:
                # if marker.id == self.base_id:
                #     state = self.base_state
                if marker.id in self.kami_ids:
                    if len(self.dones == 1):
                        if self.dones[0]:
                            continue
                        state = self.states[0]
                    else:
                        if self.dones[self.kami_ids == marker.id]:
                            continue
                        state = self.states[self.kami_ids == marker.id]
                else:
                    continue
                    # print("\nSHOULD NOT HAVE BEEN REACHED\n")
                    # raise ValueError
                
                state[0] = marker.pose.pose.position.x
                state[1] = marker.pose.pose.position.y

                o = marker.pose.pose.orientation
                o_list = [o.x, o.y, o.z, o.w]
                _, _, z = euler_from_quaternion(o_list)
                state[2] = z % (2 * np.pi)
        except:
            print(f"could not update states, id: {marker.id}")
            import pdb;pdb.set_trace()
        self.n_updates += 1

    def run(self):
        if self.n_updates == 0:
            return
        # states = self.states - self.base_state

        # one robot, start
        n_updates = self.n_updates
        current_states = []
        while len(current_states) < 5:
            if self.n_updates == n_updates:
                continue
            n_updates = self.n_updates
            current_states.append(self.states[0])
        
        current_state = np.array(current_states).mean(axis=0).squeeze()
        states = current_state[None, :]
        # one robot, end

        theta = states[:, -1]
        states = np.append(np.append(states[:, :-1], np.sin(theta)[:, None], axis=1), np.cos(theta)[:, None], axis=1)
        # states[:, 2] %= 2 * np.pi
        if not hasattr(self, "init_states"):
            self.init_states = states
        # print(states)
        diff = self.goal - states
        distances = np.linalg.norm(diff[:, :2], axis=-1)
        # tdist1 = diff[:, -1] % (2*np.pi)
        # tdist2 = 2 * np.pi - (diff[:, -1] % (2 * np.pi))
        # dists_theta = np.stack((tdist1, tdist2)).min(axis=0)
        # distances = dists_xy + 0.00 * dists_theta
        self.dists.append(distances[0])
        print("DISTANCE:", distances)
        self.dones[distances < self.tol] = True
        
        if np.all(self.dones) or self.n == 100000000000:
            action_req = RobotCmd()
            action_req.left_pwm = 0.0
            action_req.right_pwm = 0.0
            for id in self.kami_ids:
                self.command_action(action_req, f"kami{id}")

            plt.plot(np.arange(len(self.dists)), self.dists, "b-")
            plt.xlabel("Step")
            plt.ylabel("Distance to Goal")
            plt.title("Kamigami MPC")
            plt.show()
            rospy.signal_shutdown("Finished! All robots reached goal.")


        for i, agent in enumerate(self.agents):
            if not self.dones[i]:
                # perp_weight = 0.5 if distances[i] > 0.15 else 0.0
                action = agent.mpc_action(states[i], self.init_states[i], self.goal, self.state_range,
                                        self.action_range, n_steps=self.mpc_steps,
                                        n_samples=self.mpc_samples, swarm=False, swarm_weight=0.0, perp_weight=1.0,
                                        angle_weight=0.0, forward_weight=0.0, dist_weight=1.0).detach().numpy()

                action_req = RobotCmd()
                action_req.left_pwm = action[0]
                action_req.right_pwm = action[1]
                print("ACTION:", action)
                print("STATE:", states[i], '\n')
                self.command_action(action_req, f"kami{self.kami_ids[i]}")
                n_updates = self.n_updates
        rospy.sleep(0.1)
        while self.n_updates - n_updates < 10:
            # print("SLEEPING")
            rospy.sleep(0.001)
        self.n += 1


if __name__ == '__main__':
    kami_ids = [2]
    agent_path = "/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/agents/real.pkl"

    # limits: (0, 0), (1.3, 1.0)
    goal = np.array([-1.2,  -0.6, 0.0, 1.0])
    mpc_steps = 4
    mpc_samples = 500
    r = RealMPC(kami_ids, agent_path, goal, mpc_steps, mpc_samples)

    plt.plot(np.arange(len(r.dists)), r.dists, 'b-')