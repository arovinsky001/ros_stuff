#!/usr/bin/env python3

import argparse
import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from requests import head

import rospy
from ar_track_alvar_msgs.msg import AlvarMarkers
from ros_stuff.srv import CommandAction
from ros_stuff.msg import RobotCmd
from tf.transformations import euler_from_quaternion
from real_mpc_dynamics import *

SAVE_PATH = "/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/sim/data/real_data_online400.npz"

class RealMPC:
    def __init__(self, robot_ids, agent_path, goals, mpc_steps, mpc_samples, model, n_rollouts):
        self.n_rollouts = n_rollouts
        self.model = model
        self.step_count = 1
        self.n_updates = 0
        self.n_avg_states = 4
        self.n_wait_updates = 4
        self.flat_lim = 0.6
        # self.collect_data = True
        self.collect_data = False
        self.states = []
        self.actions = []
        self.next_states = []
        self.steps_to_goal = []
        self.times_to_goal = []
        self.heading_losses = []
        self.perp_losses = []
        self.heading_losses_total = []
        self.perp_losses_total = []

        self.robot_ids = np.array(robot_ids)
        self.agents = []
        self.goals = goals
        self.goal = goals[0]
        self.done_count = 0
        self.tol = 0.06
        self.dones = np.array([False] * len(robot_ids))
        self.current_states = np.zeros((len(robot_ids), 3))
        action = 0.99
        self.action_range = np.array([[-action, -action, 0.1], [action, action, 0.5]])
        self.mpc_steps = mpc_steps
        self.mpc_samples = mpc_samples
        self.dists = []
        for _ in self.robot_ids:
            with open(agent_path, "rb") as f:
                self.agents.append(pkl.load(f))
                self.agents[-1].model.eval()
        rospy.init_node('laptop_client')

        # Get info on positioning from camera & AR tags
        print("waiting for service")
        rospy.wait_for_service(f'/kami{robot_ids[0]}/server')
        self.command_action = rospy.ServiceProxy(f'/kami{robot_ids[0]}/server', CommandAction)
        print("service loaded")
        rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.update_state, queue_size=1)

        self.time = rospy.get_time()
        while not rospy.is_shutdown():
            self.run()

    def update_state(self, msg):
        try:
            found_robot = False
            for marker in msg.markers:
                if marker.id in self.robot_ids:
                    idx = np.argwhere(self.robot_ids == marker.id)
                    if len(self.dones == 1):
                        if self.dones[0]:
                            continue
                        state = self.current_states[0]
                    else:
                        if self.dones[idx]:
                            continue
                        state = self.current_states[idx]
                    found_robot = True
                else:
                    continue

                o = marker.pose.pose.orientation
                o_list = [o.x, o.y, o.z, o.w]
                x, y, z = euler_from_quaternion(o_list)

                if abs(np.sin(x)) > self.flat_lim or abs(np.sin(y)) > self.flat_lim:
                    print("MARKER NOT FLAT ENOUGH")
                    print(np.sin(x), np.sin(y))
                    return

                state[0] = marker.pose.pose.position.x
                state[1] = marker.pose.pose.position.y
                state[2] = z % (2 * np.pi)

            if found_robot:
                self.n_updates += 1
            else:
                print("not found!")
        except:
            print(f"could not update states, id: {marker.id}")
            import pdb;pdb.set_trace()
        

    def run(self):
        if self.n_updates == 0:
            return

        states = self.get_states()

        if not hasattr(self, "init_states"):
            self.init_states = states

        diff = self.goal - states
        distances = np.linalg.norm(diff[:, :2], axis=-1).squeeze()
        print("DISTANCES:", distances)
        self.dists.append(distances)
        self.dones[distances < self.tol] = True
        
        path = "/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/sim/data/loss/"
        if np.all(self.dones) or self.step_count % 40 == 0:
            self.steps_to_goal.append(self.step_count)
            self.times_to_goal.append(rospy.get_time() - self.time)
            self.heading_losses_total.append(np.sum(self.heading_losses))
            self.perp_losses_total.append(np.sum(self.perp_losses))
            self.heading_losses = []
            self.perp_losses = []
            if self.done_count == len(self.goals) - 1:
                steps_to_goal = np.array(self.steps_to_goal)
                times_to_goal = np.array(self.times_to_goal)
                perp_losses = np.array(self.perp_losses_total)
                heading_losses = np.array(self.heading_losses_total)

                n_goals = len(self.steps_to_goal) // self.n_rollouts
                steps_to_goal = steps_to_goal.reshape(-1, n_goals).sum(axis=-1)
                times_to_goal = times_to_goal.reshape(-1, n_goals).sum(axis=-1)
                perp_losses = perp_losses.reshape(-1, n_goals).sum(axis=-1)
                heading_losses = heading_losses.reshape(-1, n_goals).sum(axis=-1)
                
                assert len(steps_to_goal) == len(perp_losses) == len(heading_losses) == len(self.goals) / n_goals
                data = np.array([[steps_to_goal.mean(), steps_to_goal.std(), steps_to_goal.min(), steps_to_goal.max()],
                                 [times_to_goal.mean(), times_to_goal.std(), times_to_goal.min(), times_to_goal.max()],
                                 [perp_losses.mean(), perp_losses.std(), perp_losses.min(), perp_losses.max()],
                                 [heading_losses.mean(), heading_losses.std(), heading_losses.min(), heading_losses.max()]])

                np.save(path + f"robot{self.robot_ids[0]}_{self.model}", data)

                print("(steps, times, perp, heading), (mean, std, min, max)")
                print("DATA:", data, "\n")

                plt.plot(np.arange(len(self.dists)), self.dists, "b-")
                plt.xlabel("Step")
                plt.ylabel("Distance to Goal")
                plt.title("Kamigami MPC")
                plt.show()
                rospy.signal_shutdown("Finished! All robots reached goal.")
            else:
                for i in range(len(self.dones)):
                    self.dones[i] = False
                self.done_count += 1
                self.goal = self.goals[self.done_count]
                print(f"\n\nNEW GOAL: {self.goal}\n")
                self.init_states = self.goals[None, self.done_count-1]
                self.step_count = 1
                self.time = rospy.get_time()
                return

        for i, agent in enumerate(self.agents):
            if not self.dones[i]:
                if self.model == "joint0":
                    which = 0
                elif self.model == "joint2":
                    which = 2
                else:
                    which = None

                # for actions with duration
                swarm_weight = 0.0
                perp_weight = 1.7
                heading_weight = 0.7
                forward_weight = 0.0
                dist_weight = 0.8
                norm_weight = 0.0
                action, dist, heading, perp = agent.mpc_action(states[i], self.init_states[i], self.goal,
                                        self.action_range, swarm=False, n_steps=self.mpc_steps,
                                        n_samples=self.mpc_samples, swarm_weight=swarm_weight, perp_weight=perp_weight,
                                        heading_weight=heading_weight, forward_weight=forward_weight,
                                        dist_weight=dist_weight, norm_weight=norm_weight, which=which)
                
                action = action.detach().numpy()
                dist = dist.detach().numpy().squeeze()[0]
                heading = heading.detach().numpy().squeeze()[0]
                perp = perp.detach().numpy().squeeze()[0]

                print("\nDIST:", dist*dist_weight)
                print("HEADING:", heading*dist*heading_weight)
                print("PERP:", perp*dist*perp_weight, "\n")

                # action += np.random.normal(0.0, 0.005, size=action.shape)
                action = np.clip(action, [-0.999, -0.999, 0.001], [0.999, 0.999, 5])

                action_req = RobotCmd()
                action_req.left_pwm = action[0]
                action_req.right_pwm = action[1]
                action_req.duration = action[2]
                print("\nACTION:", action)
                state_n = states[i].copy().squeeze()
                state_n[-1] *= 180 / np.pi
                print("STATE:", state_n, '\n')
                self.command_action(action_req, f"kami{self.robot_ids[i]}")

                self.heading_losses.append(heading)
                self.perp_losses.append(perp)

        n_updates = self.n_updates
        while self.n_updates - n_updates < self.n_wait_updates:
            rospy.sleep(0.001)

        if self.collect_data:
            next_states = self.get_states()
            self.step_count += 1
            self.states.append(states[0])
            self.actions.append(action)
            self.next_states.append(next_states[0])
            if self.step_count % 5 == 0:
                self.save_data()
        else:
            self.step_count += 1
        
        # if self.online:
            # next_states = []
            # while len(next_states) < self.n_avg_states:
            #     if self.n_updates == n_updates:
            #         continue
            #     n_updates = self.n_updates
            #     next_states.append(self.states[0].copy())
            
            # next_states = np.array(next_states).mean(axis=0).squeeze()[None, :]
            # theta = next_states[:, -1]
            # next_states = np.append(np.append(next_states[:, :-1], np.sin(theta)[:, None], axis=1), np.cos(theta)[:, None], axis=1)
            
            # states = (states - self.agents[0].states_mean) / self.agents[0].states_std
            # action = (action - self.agents[0].actions_mean) / self.agents[0].actions_std
            # next_states = (next_states - self.agents[0].next_states_mean) / self.agents[0].next_states_std
            # self.agents[0].model.update(states, action[None, :], next_states)
    
    def get_states(self):
        if self.n_avg_states > 1:
            current_states = []
            n_updates = self.n_updates
            while len(current_states) < self.n_avg_states:
                if self.n_updates == n_updates:
                    rospy.sleep(0.001)
                n_updates = self.n_updates
                current_states.append(self.current_states.copy())
            
            current_states = np.array(current_states).squeeze()
            if len(current_states.shape) == 2:
                current_states = current_states.mean(axis=0, keepdims=True)
            else:
                current_states = current_states.mean(axis=1)

            return current_states
        else:
            return self.current_states.copy()
    
    def save_data(self, clip_end=False):
        states = np.array(self.states)
        actions = np.array(self.actions)
        next_states = np.array(self.next_states)

        if clip_end:
            clip = 3
            states = states[:-clip]
            actions = actions[:-clip]
            next_states = next_states[:-clip]
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
    parser = argparse.ArgumentParser(description='Do MPC.')
    parser.add_argument('-robot_ids', nargs='+', help='robot ids for rollout', type=int)
    parser.add_argument('-model', type=str, help='model to use for experiment')
    parser.add_argument('-mpc_steps', type=int)
    parser.add_argument('-mpc_samples', type=int)
    parser.add_argument('-n_rollouts', type=int)

    args = parser.parse_args()

    agent_path = "/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/agents/"
    if "joint" in args.model:
        agent_path += "real_multi.pkl"
    elif "single0" in args.model:
        agent_path += "real_single0.pkl"
    elif "single2" in args.model:
        agent_path += "real_single2.pkl"
    elif "naive" in args.model:
        agent_path += "real_multi_naive.pkl"
    elif "ded" in args.model:
        _, n1, n2 = args.model.split("_")
        agent_path += f"real_single{n1}_retrain{n2}.pkl"
    elif "100" in args.model:
        agent_path += "real_single2_100.pkl"
    elif "200" in args.model:
        agent_path += "real_single2_200.pkl"
    elif "400" in args.model:
        agent_path += "real_single2_400.pkl"
    else:
        raise ValueError
    # agent_path = "/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/agents/real_AMAZING_kami1.pkl"

    # xgoals = np.linspace(-1.3, -0.5, 10)
    # ygoals = np.linspace(-0.9, 0.0, 10)
    # traj1 = np.block([xgoals.reshape(-1, 1), np.tile(ygoals[-1], (len(xgoals), 1))])
    # traj2 = np.block([np.tile(xgoals[-1], (len(ygoals), 1)), np.flip(ygoals).reshape(-1, 1)])
    # traj3 = np.block([np.flip(xgoals).reshape(-1, 1), np.tile(ygoals[0], (len(xgoals), 1))])
    # traj4 = np.block([np.tile(xgoals[0], (len(ygoals), 1)), ygoals.reshape(-1, 1)])
    # goals = np.append(np.block([[traj1], [traj2], [traj3], [traj4]]), np.zeros(((len(xgoals) + len(ygoals)) * 2, 1)), axis=-1)
    
    # goals = np.array([[-0.45, -0.05],
    #                   [-0.45, -0.75],
    #                   [-1.3, -0.75],
    #                   [-1.3, -0.05]])

    goals = np.array([[-1.0, 0.0],
                      [-0.6, -0.1],
                      [-0.5, -0.3],
                      [-0.7, -0.45],
                      [-1.0, -0.5],
                      [-1.3, -0.6],
                      [-1.3, -0.9],
                      [-0.95, -0.95],
                      [-0.65, -0.8],
                      [-0.5, -0.6],
                      [-0.7, -0.4],
                      [-1.0, -0.25],
                      [-1.3, -0.05]])

    goals = np.append(goals, np.zeros((len(goals), 1)), axis=1)

    # goals = np.array([[-0.5, -0.6, 0.0],
    #                   [-1.0, -0.8, 0.0],
    #                   [-1.4, 0.0, 0.0]])
    
    goals = np.tile(goals, (args.n_rollouts, 1))

    # n_goals = 200
    # goals = np.random.uniform(low=[-1.4, -1.0], high=[-0.4, 0.0], size=(n_goals, 2))
    # goals = np.append(goals, np.tile(np.array([0., 1.]), (n_goals, 1)), axis=-1)

    r = RealMPC(args.robot_ids, agent_path, goals, args.mpc_steps, args.mpc_samples, args.model, args.n_rollouts)
