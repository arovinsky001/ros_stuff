#!/usr/bin/python3

import argparse
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import rospy

from real_mpc_dynamics import *
from utils import KamigamiInterface
from ros_stuff.msg import RobotCmd

from tf.transformations import euler_from_quaternion

SAVE_PATH = "/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/sim/data/real_data_online400.npz"

class RealMPC(KamigamiInterface):
    def __init__(self, robot_id, object_id, agent_path, mpc_steps, mpc_samples, model, n_rollouts, tolerance, lap_time, collect_data, calibrate, plot, new_buffer):
        self.halted = False
        self.object_id = object_id
        self.object_state = np.zeros(3)    # (x, y, theta)

        super().__init__([robot_id], SAVE_PATH, calibrate, new_buffer=new_buffer)
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
        # self.swarm_weight = 0.0
        # self.perp_weight = 4.
        # self.heading_weight = 0.8
        # self.dist_weight = 3.0
        # self.norm_weight = 0.0
        # self.dist_bonus_factor = 10.

        self.swarm_weight = 0.0
        self.perp_weight = 0.
        self.heading_weight = 0.
        self.dist_weight = 1.0
        self.norm_weight = 0.0
        self.dist_bonus_factor = 0.

        self.plot_states = []
        self.plot_goals = []

        buffer_size = 1000
        self.stamped_losses = np.zeros((buffer_size, 5))      # timestamp, dist, heading, perp, total
        self.losses = np.empty((0, 4))

        self.time_elapsed = 0.
        self.logged_transitions = 0
        self.laps = 0
        self.n_prints = 0

        # with open(agent_path, "rb") as f:
        #     self.agent = pkl.load(f)
        input_dim = 8       # 4
        output_dim = 8      # 4
        self.agent = MPCAgent(input_dim, output_dim, seed=0, dist=False,
                         scale=True, multi=False, hidden_dim=500,
                         hidden_depth=4, lr=0.001,
                         dropout=0.3, entropy_weight=0.0, ensemble=1)
        
        idx = self.replay_buffer.capacity if self.replay_buffer.full else self.replay_buffer.idx
        states = self.replay_buffer.states[:idx]
        actions = self.replay_buffer.actions[:idx]
        next_states = self.replay_buffer.next_states[:idx]
        self.agent.train(states, actions, next_states, set_scalers=True, epochs=100, batch_size=1000, use_all_data=True)
        
        for g in self.agent.models[0].optimizer.param_groups:
            g['lr'] = 1e-4

        np.set_printoptions(suppress=True)

    def define_goal_trajectory(self):
        # self.front_left_corner = np.array([-0.3, -1.0])
        # self.back_right_corner = np.array([-1.45, 0.0])
        self.front_left_corner = np.array([-0.1, -1.15])
        self.back_right_corner = np.array([-1.9, 0.1])
        corner_range = self.back_right_corner - self.front_left_corner

        # radius_rel = 0.3
        back_circle_center_rel = np.array([0.38, 0.65])
        front_circle_center_rel = np.array([0.74, 0.3])
        
        self.back_circle_center = back_circle_center_rel * corner_range + self.front_left_corner
        self.front_circle_center = front_circle_center_rel * corner_range + + self.front_left_corner
        # self.radius = np.abs(corner_range).mean() * radius_rel
        self.radius = np.linalg.norm(self.back_circle_center - self.front_circle_center) / 2

    def update_states(self, msg):
        if self.halted:
            return
        # update object state
        object_found = False
        for marker in msg.markers:
            if marker.id == self.object_id:
                o = marker.pose.pose.orientation
                o_list = [o.x, o.y, o.z, o.w]
                x, y, z = euler_from_quaternion(o_list)

                flat_lim = 0.7
                if abs(np.sin(x)) > flat_lim or abs(np.sin(y)) > flat_lim and self.started:
                    print("OBJECT MARKER NOT FLAT ENOUGH")
                    print("sin(x):", np.sin(x), "|| sin(y)", np.sin(y))
                    self.not_found = True
                    return
                
                if hasattr(self, "tag_offsets"):
                    z += self.tag_offsets[marker.id]

                self.object_state[0] = marker.pose.pose.position.x
                self.object_state[1] = marker.pose.pose.position.y
                self.object_state[2] = z % (2 * np.pi)
                object_found = True
                break

        if not object_found:
            self.not_found = True
            print("OBJECT NOT FOUND")
            return

        super().update_states(msg)
        if self.not_found:
            return

        state = np.concatenate((self.current_states.squeeze()[:-1], self.object_state), axis=0)
        last_goal = self.last_goal if self.started else self.object_state
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

            if not self.step():
                self.halted = True
                import pdb;pdb.set_trace()
                self.halted = False

            if self.done:
                rospy.signal_shutdown("Finished! All robots reached goal.")
                return

    def dist_to_start(self):
        object_state = self.get_states(wait=False).squeeze()[3:]
        return np.linalg.norm((object_state - self.init_goal)[:2])

    def step(self):
        state = self.get_states()
        if state is None:
            print("MARKERS NOT VISIBLE")
            return False

        action = self.get_take_actions(state)
        if action is None:
            print("MARKERS NOT VISIBLE")
            return False

        next_state = self.get_states()
        if next_state is None:
            print("MARKERS NOT VISIBLE")
            return False

        self.collect_training_data(state, action, next_state)
        self.update_model_online()
        self.check_rollout_finished()

        return True

    def get_states(self, perturb=False, wait=True):
        robot_state = super().get_states(perturb=perturb, wait=wait).squeeze()[:-1]

        time = rospy.get_time()
        if self.n_avg_states > 1:
            object_state = []
            while len(object_state) < self.n_avg_states:
                if rospy.get_time() - time > 3:
                    return None
                if wait and self.n_updates == self.last_n_updates:
                    rospy.sleep(0.001)
                    continue
                self.last_n_updates = self.n_updates
                object_state.append(self.object_state.copy())
            
            object_state = np.array(object_state).squeeze().mean(axis=0)
        else:
            while wait and self.n_updates == self.last_n_updates:
                if rospy.get_time() - time > 3:
                    return None
                rospy.sleep(0.001)
            self.last_n_updates = self.n_updates

            object_state = self.object_state.copy().squeeze()
        
        states = np.concatenate((robot_state, object_state), axis=-1)
        return states

    def get_take_actions(self, state):
        if self.model == "joint0":
            which = 0
        elif self.model == "joint2":
            which = 2
        else:
            which = None
        
        goal = self.get_goal()
        object_state = state[3:]
        last_goal = self.last_goal if self.started else object_state
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
        self.remap_cmd(action_req, self.robot_id)

        self.service_proxies[0](action_req, f"kami{self.robot_id}")
        self.time_elapsed += action_req.duration if self.started else 0

        time = rospy.get_time()
        bool_idx = (self.stamped_losses[:, 0] > time - action_req.duration) & (self.stamped_losses[:, 0] < time)
        idx = np.argwhere(bool_idx).squeeze().reshape(-1)

        self.n_prints += 1
        print(f"\n\n\n\nNO. {self.n_prints}")
        print("/////////////////////////////////////////////////")
        print("=================================================")
        print(f"RECORDING {len(idx)} LOSSES\n")
        print("GOAL:", goal)
        print("STATE:", state)
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
        time = rospy.get_time()
        while self.n_updates - n_updates < self.n_wait_updates:
            if rospy.get_time() - time > 3:
                return None
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
            # Print current cumulative loss per lap completed
            dist_losses, heading_losses, perp_losses, total_losses = self.losses.T
            data = np.array([[dist_losses.mean(), dist_losses.std(), dist_losses.min(), dist_losses.max()],
                         [perp_losses.mean(), perp_losses.std(), perp_losses.min(), perp_losses.max()],
                         [heading_losses.mean(), heading_losses.std(), heading_losses.min(), heading_losses.max()],
                         [total_losses.mean(), total_losses.std(), total_losses.min(), total_losses.max()]])
            print("lap:", self.laps)
            print("rows: (dist, perp, heading, total)")
            print("cols: (mean, std, min, max)")
            print("DATA:", data, "\n")
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

    def collect_training_data(self, state, action, next_state):
        # if self.started and self.collect_data:
        if self.collect_data:
            self.replay_buffer.add(state, action, next_state)

            if self.replay_buffer.idx % self.save_freq == 0:
                print(f"\nSAVING REPLAY BUFFER WITH {self.replay_buffer.idx} TRANSITIONS\n")
                with open("/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/replay_buffers/buffer.pkl", "wb") as f:
                    pkl.dump(self.replay_buffer, f)

            # self.states.append(state)
            # self.actions.append(action)
            # self.next_states.append(next_state)
            # self.logged_transitions += 1
            # if self.logged_transitions % self.save_freq == 0:
            #     self.save_training_data()
    
    # currently only supports one robot
    def update_model_online(self):
        if self.replay_buffer.full or self.replay_buffer.idx > 10:
            # sample from buffer
            states, actions, next_states = self.replay_buffer.sample(200)
            states, states_delta = self.agent.convert_state_delta(states, next_states)

            # # scale
            # states, actions, _ = self.agent.get_scaled(states, actions, None)
            # states_delta = self.agent.get_scaled(states_delta)

            # take single gradient step
            for model in self.agent.models:
                model.update(states, actions, states_delta)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do MPC.')
    parser.add_argument('-robot_id', type=int, help='robot id for rollout')
    parser.add_argument('-object_id', type=int, help='object id for rollout')
    parser.add_argument('-model', type=str, help='model to use for experiment')
    parser.add_argument('-mpc_steps', type=int)
    parser.add_argument('-mpc_samples', type=int)
    parser.add_argument('-n_rollouts', type=int)
    parser.add_argument('-tolerance', type=float, default=0.05)
    parser.add_argument('-collect_data', action='store_true')
    parser.add_argument('-lap_time', type=float)
    parser.add_argument('-calibrate', action='store_true')
    parser.add_argument('-plot', action='store_true')
    parser.add_argument('-new_buffer', action='store_true')

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

    r = RealMPC(args.robot_id, args.object_id, agent_path, args.mpc_steps, args.mpc_samples, args.model, args.n_rollouts, args.tolerance, args.lap_time, args.collect_data, args.calibrate, args.plot, args.new_buffer)
    r.run()
