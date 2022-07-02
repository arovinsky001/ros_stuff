#!/usr/bin/python3

import argparse
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import rospy

from ros_stuff.msg import RobotCmd
from real_mpc_dynamics import *
from utils import KamigamiInterface

SAVE_PATH = "/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/sim/data/real_data_online400.npz"

class RealMPC(KamigamiInterface):
    def __init__(self, robot_id, agent_path, mpc_steps, mpc_samples, model, n_rollouts, tolerance, lap_time, collect_data):
        super().__init__([robot_id], SAVE_PATH)
        self.define_goal_trajectory()

        self.mpc_steps = mpc_steps
        self.mpc_samples = mpc_samples
        self.model = model
        self.n_rollouts = n_rollouts
        self.tolerance = tolerance
        self.lap_time = lap_time
        self.collect_data = collect_data
        self.robot_id = self.robot_ids[0]

        # weights for MPC cost terms
        self.swarm_weight = 0.0
        self.perp_weight = 0.0
        self.heading_weight = 0.0
        self.forward_weight = 0.0
        self.dist_weight = 1.0
        self.norm_weight = 0.0

        self.plot_states = []
        self.plot_goals = []

        buffer_size = 1000
        self.stamped_losses = np.zeros(buffer_size, 5)      # timestamp, dist, heading, perp, total
        self.losses = np.empty((0, 4))

        self.time_elapsed = 0.
        self.logged_transitions = 0
        self.laps = 0
        self.started = False
        
        with open(agent_path, "rb") as f:
            self.agent = pkl.load(f)
            self.agent.model.eval()

    def define_goal_trajectory(self):
        self.front_left_corner = np.array([-0.4, -0.9])
        self.back_right_corner = np.array([-1.4, 0.1])
        range = self.back_right_corner - self.front_left_corner

        radius_rel = 0.3
        back_circle_center_rel = np.array([0.38, 0.65])
        front_circle_center_rel = np.array([0.74, 0.3])
        
        self.back_circle_center = back_circle_center_rel * range + self.front_left_corner
        self.front_circle_center = front_circle_center_rel * range + + self.front_left_corner
        self.radius = np.abs(range).mean() * radius_rel
        self.lap_time = 40
        self.last_goal = self.back_circle_center + np.array([0.0, 1.0]) * self.radius
        self.last_goal = np.block([self.last_goal, 0.0])
    
    def update_states(self, msg):
        super().update_states(msg)
        if self.not_found:
            return

        state = self.current_states.squeeze()
        dist_loss, heading_loss, perp_loss = self.agent.compute_losses(state, self.prev_goal, self.get_goal(), self.n_samples, logging=True)
        total_loss = dist_loss * self.dist_weight + heading_loss * self.heading_weight + perp_loss * self.perp_weight
        
        self.losses[1:] = self.losses[:-1]
        self.losses[0] = [rospy.get_time(), dist_loss, heading_loss, perp_loss, total_loss]

        print("\nDIST:", dist_loss * self.dist_weight)
        print("HEADING:", heading_loss * dist_loss * self.heading_weight)
        print("PERP:", perp_loss * dist_loss * self.perp_weight, "\n")
    
    def run(self):
        while self.n_updates == 0:
            print("WAITING FOR FIRST AR TRACKING UPDATE")
            rospy.sleep(0.1)

        first_plot = True
        self.time_elapsed = self.action_range[0, -1]
        self.init_goal = self.get_goal()
        while not rospy.is_shutdown():
            self.step()

            if self.done:
                rospy.signal_shutdown("Finished! All robots reached goal.")
                return
            
            if self.dist_to_start() < self.tolerance:
                self.started = True

            plot_goals = np.array(self.plot_goals)
            plot_states = np.array(self.plot_states)
            plt.plot(plot_goals[:, 0], plot_goals[:, 1], color="green", linewidth=1.5, marker="*", label="Goal Trajectory")
            plt.plot(plot_states[:, 0], plot_states[:, 1], color="red", linewidth=1.5, marker=">", label="Actual Trajectory")
            if first_plot:
                plt.legend()
                plt.ion()
                plt.show()
                first_plot = False
            plt.draw()
            plt.pause(0.001)

    def dist_to_start(self):
        return np.linalg.norm((self.get_states().squeeze()[:-1] - self.init_goal)[:2])

    def step(self):
        state = self.get_states().squeeze()
        action = self.get_take_actions(state[:-1])

        self.collect_training_data(state, action)
        self.check_rollout_finished()
    
    def get_take_actions(self, state):
        if self.model == "joint0":
            which = 0
        elif self.model == "joint2":
            which = 2
        else:
            which = None
        
        goal = self.get_goal()
        last_goal = self.last_goal if self.started else state
        action = self.agent.mpc_action(state, last_goal, goal,
                                self.action_range, swarm=False, n_steps=self.mpc_steps,
                                n_samples=self.mpc_samples, swarm_weight=self.swarm_weight, perp_weight=self.perp_weight,
                                heading_weight=self.heading_weight, forward_weight=self.forward_weight,
                                dist_weight=self.dist_weight, norm_weight=self.norm_weight, which=which)
        
        self.last_goal = goal.copy() if self.started else self.last_goal

        action = np.clip(action, *self.action_range)
        action_req = RobotCmd()
        action_req.left_pwm = action[0]
        action_req.right_pwm = action[1]
        action_req.duration = action[2]
        print("\nACTION:", action)

        state_n = state.copy()
        state_n[-1] *= 180 / np.pi
        print("STATE (x, y, theta [deg]):", state_n, '\n')
        self.service_proxies[0](action_req, f"kami{self.robot_id}")
        self.time_elapsed += action_req.duration if self.started else 0

        time = rospy.get_time()
        bool_idx = (self.stamped_losses[:, 0] > time - action_req.duration) & (self.stamped_losses[:, 0] < time)
        idx = np.argwhere(bool_idx).squeeze()
        self.losses = np.append(self.losses, self.stamped_losses[idx, 1:], axis=0)

        n_updates = self.n_updates
        while self.n_updates - n_updates < self.n_wait_updates:
            rospy.sleep(0.001)

        return action

    def get_goal(self):
        t_rel = (self.time_elapsed % self.lap_time) / self.lap_time

        if t_rel < 0.25:
            theta = 2 * np.pi * t_rel / 0.5
            center = self.back_circle_center
        elif 0.25 <= t_rel < 0.75:
            theta = -2 * np.pi * (t_rel - 0.25) / 0.5
            center = self.front_circle_center
        else:
            theta = 2 * np.pi * (t_rel - 0.5) / 0.5
            center = self.back_circle_center
        
        theta += np.pi / 4
        goal = center + np.array([np.sin(theta), np.cos(theta)]) * self.radius
        
        print("GOAL:", goal)
        self.plot_goals.append(goal)

        return np.block([goal, 0.0])
    
    def check_rollout_finished(self):
        if self.time_elapsed > self.lap_time and self.dist_to_start() < self.tolerance:
            self.laps += 1
            self.time_elapsed = 0.
        
        if self.laps == self.n_rollouts:
            self.dump_performance_metrics()
            self.done = True
            
    def dump_performance_metrics(self):
        path = "/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/sim/data/loss/"
        np.save(path + f"robot{self.robot_id}_{self.model}", self.losses)

        dist_losses, heading_losses, perp_losses, total_losses = self.losses.T
        data = np.array([[dist_losses.mean(), dist_losses.std(), dist_losses.min(), dist_losses.max()],
                         [perp_losses.mean(), perp_losses.std(), perp_losses.min(), perp_losses.max()],
                         [heading_losses.mean(), heading_losses.std(), heading_losses.min(), heading_losses.max()],
                         [total_losses.mean(), total_losses.std(), total_losses.min(), total_losses.max()]])

        print("rows: (dist, perp, heading, total)")
        print("cols: (mean, std, min, max)")
        print("DATA:", data, "\n")

    def collect_training_data(self, state, action):
        if self.started:
            self.plot_states.append(state)
            if self.collect_data:
                next_state = self.get_states()
                self.states.append(state)
                self.actions.append(action)
                self.next_states.append(next_state)
                self.logged_transitions += 1
                if self.logged_transitions % self.save_freq == 0:
                    self.save_training_data()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do MPC.')
    parser.add_argument('-robot_id', type=int, help='robot id for rollout')
    parser.add_argument('-model', type=str, help='model to use for experiment')
    parser.add_argument('-mpc_steps', type=int)
    parser.add_argument('-mpc_samples', type=int)
    parser.add_argument('-n_rollouts', type=int)
    parser.add_argument('-tolerance', type=float, default=0.05)
    parser.add_argument('-collect_data', action='store_true')
    parser.add_argument('-lap_time', type=float)

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
    elif "amazing" in args.model:
        agent_path += "real_AMAZING_kami1.pkl"
    else:
        raise ValueError

    r = RealMPC(args.robot_id, agent_path, args.mpc_steps, args.mpc_samples, args.model, args.n_rollouts, args.tolerance, args.lap_time, args.collect_data)
    r.run()
