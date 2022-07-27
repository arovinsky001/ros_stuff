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
    def __init__(self, robot_id, agent_path, mpc_steps, mpc_samples, model, n_rollouts, tolerance, lap_time, collect_data, calibrate, plot):
        with open(agent_path, "rb") as f:
            self.agent = pkl.load(f)
        super().__init__([robot_id], SAVE_PATH, calibrate)
        self.define_goal_trajectory()

        self.mpc_steps = mpc_steps
        self.mpc_samples = mpc_samples
        self.model = model
        self.n_rollouts = n_rollouts
        self.tolerance = tolerance
        self.lap_time = lap_time
        self.collect_data = collect_data
        self.plot = plot
        self.robot_id = self.robot_ids[0]

        # weights for MPC cost terms
        self.swarm_weight = 0.0
        # self.perp_weight = 15.
        # self.heading_weight = 0.5
        # self.dist_weight = 50.
        self.perp_weight = 4.
        self.heading_weight = 0.8
        self.dist_weight = 3.0
        self.norm_weight = 0.0
        self.dist_bonus_factor = 10.

        # self.swarm_weight = 0.0
        # self.perp_weight = 2.0
        # self.heading_weight = 0.01
        # self.dist_weight = 20.
        # self.norm_weight = 0.0

        self.plot_states = []
        self.plot_goals = []

        buffer_size = 1000
        self.stamped_losses = np.zeros((buffer_size, 5))      # timestamp, dist, heading, perp, total
        self.losses = np.empty((0, 4))

        self.time_elapsed = 0.
        self.logged_transitions = 0
        self.laps = 0
        self.n_prints = 0
        
        np.set_printoptions(suppress=True)

    def define_goal_trajectory(self):
        self.front_left_corner = np.array([-0.3, -1.0])
        self.back_right_corner = np.array([-1.45, 0.0])
        corner_range = self.back_right_corner - self.front_left_corner

        # radius_rel = 0.3
        back_circle_center_rel = np.array([0.38, 0.65])
        front_circle_center_rel = np.array([0.74, 0.3])
        
        self.back_circle_center = back_circle_center_rel * corner_range + self.front_left_corner
        self.front_circle_center = front_circle_center_rel * corner_range + + self.front_left_corner
        # self.radius = np.abs(corner_range).mean() * radius_rel
        self.radius = np.linalg.norm(self.back_circle_center - self.front_circle_center) / 2
    
    def update_states(self, msg):
        super().update_states(msg)
        if self.not_found:
            return

        state = self.current_states.squeeze()[:-1]
        last_goal = self.last_goal if self.started else state
        dist_loss, heading_loss, perp_loss = self.agent.compute_losses(state, last_goal, self.get_goal(), current=True)
        total_loss = dist_loss * self.dist_weight + heading_loss * self.heading_weight + perp_loss * self.perp_weight
        
        if self.started:
            self.stamped_losses[1:] = self.stamped_losses[:-1]
            self.stamped_losses[0] = [rospy.get_time()] + [i.detach().numpy() for i in [dist_loss, heading_loss, perp_loss, total_loss]]
    
    def run(self):
        rospy.sleep(0.2)
        while self.n_updates < 1 and not rospy.is_shutdown():
            print(f"FILLING LOSS BUFFER, {self.n_updates} UPDATES")
            rospy.sleep(0.2)

        self.first_plot = True
        self.init_goal = self.get_goal()
        while not rospy.is_shutdown():
            if self.started:
                self.plot_states.append(self.current_states.squeeze().copy())
                self.plot_goals.append(self.last_goal.copy())
                if self.plot:
                    plot_goals = np.array(self.plot_goals)
                    plot_states = np.array(self.plot_states)
                    plt.plot(plot_goals[:, 0] * -1, plot_goals[:, 1], color="green", linewidth=1.5, marker="*", label="Goal Trajectory")
                    plt.plot(plot_states[:, 0] * -1, plot_states[:, 1], color="red", linewidth=1.5, marker=">", label="Actual Trajectory")
                    if self.first_plot:
                        plt.legend()
                        plt.ion()
                        plt.show()
                        self.first_plot = False
                    plt.draw()
                    plt.pause(0.0001)

            if self.dist_to_start() < self.tolerance and not self.started:
                self.started = True
                self.last_goal = self.get_goal()
                self.time_elapsed = self.duration / 2

            self.step()

            if self.done:
                rospy.signal_shutdown("Finished! All robots reached goal.")
                return

    def dist_to_start(self):
        return np.linalg.norm((self.get_states(wait=False).squeeze()[:-1] - self.init_goal)[:2])

    def step(self):
        state = self.get_states(wait=self.started).squeeze()
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
        if "control" in self.model:
            action = self.differential_drive(state, last_goal, goal)
        else:
            action = self.agent.mpc_action(state, last_goal, goal,
                                    self.action_range, swarm=False, n_steps=self.mpc_steps,
                                    n_samples=self.mpc_samples, swarm_weight=self.swarm_weight,
                                    perp_weight=self.perp_weight, heading_weight=self.heading_weight,
                                    dist_weight=self.dist_weight, norm_weight=self.norm_weight, dist_bonus_factor=self.dist_bonus_factor,
                                    which=which).detach().numpy()

        action = np.clip(action, *self.action_range)
        action_req = RobotCmd()
        action_req.left_pwm = action[0]
        action_req.right_pwm = action[1]
        action_req.duration = self.duration
        action_req = self.remap_cmd(action_req, self.robot_id)

        self.service_proxies[0](action_req, f"kami{self.robot_id}")
        self.time_elapsed += action_req.duration if self.started else 0

        time = rospy.get_time()
        bool_idx = (self.stamped_losses[:, 0] > time - action_req.duration) & (self.stamped_losses[:, 0] < time)
        idx = np.argwhere(bool_idx).squeeze().reshape(-1)

        state_n = state.copy()
        state_n[-1] *= 180 / np.pi
        self.n_prints += 1
        print(f"\n\n\n\nNO. {self.n_prints}")
        print("/////////////////////////////////////////////////")
        print("=================================================")
        print(f"RECORDING {len(idx)} LOSSES\n")
        print("GOAL:", goal)
        print("STATE:", state_n)
        print("ACTION:", action, "\n")
        if len(idx) != 0:
            losses_to_record = self.stamped_losses[idx, 1:].squeeze()
            losses_to_record = losses_to_record[None, :] if len(losses_to_record.shape) == 1 else losses_to_record
            self.losses = np.append(self.losses, losses_to_record, axis=0)
            
            dist_loss, heading_loss, perp_loss, total_loss = losses_to_record if len(losses_to_record.shape) == 1 else losses_to_record[-1]
            print("DIST:", dist_loss)
            print("HEADING:", heading_loss)
            print("PERP:", perp_loss)
            print("TOTAL:", total_loss)
        print("=================================================")
        print("/////////////////////////////////////////////////")

        self.last_goal = goal.copy() if self.started else None
        n_updates = self.n_updates
        while self.n_updates - n_updates < self.n_wait_updates:
            rospy.sleep(0.001)

        return action
    
    def differential_drive(self, state, last_goal, goal):
        dist_loss, heading_loss, perp_loss = self.agent.compute_losses(state, last_goal, goal, current=True, signed=True)
        dist_loss, heading_loss, perp_loss = [i.detach().numpy() for i in [dist_loss, heading_loss, perp_loss]]
        ctrl_array = np.array([[0.5, 0.5], [0.5, -0.5]])
        print("DIST: ", dist_loss)
        print("HEAD: ", heading_loss)
        error_array = np.array([dist_loss * 15.0, (perp_loss * 15.0 + heading_loss * 4.0)]) * 1.0
        left_pwm, right_pwm = ctrl_array @ error_array
        return np.array([left_pwm, right_pwm])

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
        
        return np.block([goal, 0.0])
    
    def check_rollout_finished(self):
        if self.time_elapsed > self.lap_time:
            self.laps += 1
            self.time_elapsed = 0.
            
            self.started = False
            plot_goals = np.array(self.plot_goals)
            plot_states = np.array(self.plot_states)
            plt.plot(plot_goals[:, 0] * -1, plot_goals[:, 1], color="green", linewidth=1.5, marker="*", label="Goal Trajectory")
            plt.plot(plot_states[:, 0] * -1, plot_states[:, 1], color="red", linewidth=1.5, marker=">", label="Actual Trajectory")
            if self.first_plot:
                plt.legend()
                plt.ion()
                plt.show()
                self.first_plot = False
            plt.draw()
            plt.pause(0.001)
        
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
        if self.started and self.collect_data:
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
    parser.add_argument('-calibrate', action='store_true')
    parser.add_argument('-plot', action='store_true')

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
    elif "control" in args.model:
        agent_path += "real_single2.pkl"
    else:
        raise ValueError

    r = RealMPC(args.robot_id, agent_path, args.mpc_steps, args.mpc_samples, args.model, args.n_rollouts, args.tolerance, args.lap_time, args.collect_data, args.calibrate, args.plot)
    r.run()
