#!/usr/bin/python3

import os
import argparse
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import rospy

from real_mpc_dynamics import *
from replay_buffer import ReplayBuffer

from ros_stuff.msg import RobotCmd
from ros_stuff.srv import CommandAction

from ar_track_alvar_msgs.msg import AlvarMarkers
from tf.transformations import euler_from_quaternion

SAVE_PATH = "/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/sim/data/real_data_online400.npz"

class RealMPC():
    def __init__(self, robot_id, object_id, mpc_steps, mpc_samples, n_rollouts, tolerance, lap_time, calibrate, plot,
                 new_buffer, pretrain, robot_goals, scale, mpc_softmax):
        self.halted = False
        self.initialized = False
        self.epsilon = 0.0
        self.steps = 0
        self.gradient_steps = 2
        self.random_steps = 0 if pretrain else 100

        self.save_path = SAVE_PATH
        self.robot_id = robot_id
        self.object_id = object_id
        self.mpc_softmax = mpc_softmax
        self.use_object = (self.object_id >= 0)

        max_pwm = 0.999
        self.action_range = np.array([[-max_pwm, -max_pwm], [max_pwm, max_pwm]])
        # self.duration = 0.5
        self.duration = 0.2
        self.robot_state = np.zeros(3)      # (x, y, theta)
        self.object_state = np.zeros(3)     # (x, y, theta)

        self.n_updates = 0
        self.last_n_updates = 0
        self.not_found = False
        self.started = False
        self.done = False

        self.n_avg_states = 1
        self.n_wait_updates = 1
        self.n_clip = 3
        self.flat_lim = 0.6
        self.save_freq = 100000 # 50

        if os.path.exists("/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/replay_buffers/buffer.pkl") and not new_buffer:
            with open("/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/replay_buffers/buffer.pkl", "rb") as f:
                self.replay_buffer = pkl.load(f)
        else:
            state_dim = 6 if self.use_object else 3
            self.replay_buffer = ReplayBuffer(capacity=10000, state_dim=state_dim, action_dim=2)

        # rospy.init_node("laptop_client_mpc")

        # robot_id = self.robot_id if False else 0
        # print(f"waiting for robot {robot_id} service")
        # rospy.wait_for_service(f"/kami{robot_id}/server")
        # self.service_proxy = rospy.ServiceProxy(f"/kami{robot_id}/server", CommandAction)
        # print("connected to robot service")

        # print("waiting for /ar_pose_marker rostopic")
        # rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.update_state, queue_size=1)
        # print("subscribed to /ar_pose_marker")

        self.tag_offset_path = "/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/sim/data/tag_offsets.npy"
        if not os.path.exists(self.tag_offset_path) or calibrate:
            self.calibrating = True
            self.calibrate()

        self.calibrating = False
        self.tag_offsets = np.load(self.tag_offset_path)

        self.define_goal_trajectory()

        self.mpc_steps = mpc_steps
        self.mpc_samples = mpc_samples
        self.n_rollouts = n_rollouts
        self.tolerance = tolerance
        self.lap_time = lap_time
        self.plot = plot
        self.robot_goals = robot_goals
        self.scale = scale
        self.pretrain = pretrain

        # weights for MPC cost terms
        # self.perp_weight = 4.
        # self.heading_weight = 0.8
        # self.dist_weight = 3.0
        # self.norm_weight = 0.0
        # self.dist_bonus_weight = 10.
        # self.sep_weight = 0.0
        # self.discrim_weight = 0.
        # self.heading_diff_weight = 0.0

        self.perp_weight = 1.
        self.heading_weight = 0.1
        self.dist_weight = 1.
        self.norm_weight = 0.0
        self.dist_bonus_weight = 0.
        self.sep_weight = 0.
        self.discrim_weight = 0.
        self.heading_diff_weight = 0.0

        self.plot_robot_states = []
        self.plot_object_states = []
        self.plot_goals = []

        loss_buffer_size = 1000
        self.stamped_losses = np.zeros((loss_buffer_size, 5))      # timestamp, dist, heading, perp, total
        self.losses = np.empty((0, 4))

        self.time_elapsed = 0.
        self.logged_transitions = 0
        self.laps = 0
        self.n_prints = 0

        self.agent = MPCAgent(seed=0, dist=True, scale=self.scale, hidden_dim=250, hidden_depth=2,
                              lr=0.001, dropout=0.0, ensemble=1, use_object=self.use_object)

        if pretrain:
            self.train_model()

        for g in self.agent.models[0].optimizer.param_groups:
            g['lr'] = 3e-4

        np.set_printoptions(suppress=True)
        self.initialized = True

    def calibrate(self):
        tag_offsets = np.zeros(10)

        input(f"Place robot/object on the left calibration point, aligned with the calibration line and hit enter.")
        left_state = self.get_state(wait=False)
        input(f"Place robot/object on the right calibration point, aligned with the calibration line and hit enter.")
        right_state = self.get_state(wait=False)

        robot_left_state, robot_right_state = left_state[:3], right_state[:3]
        true_robot_vector = (robot_left_state - robot_right_state)[:2]
        true_robot_angle = np.arctan2(true_robot_vector[1], true_robot_vector[0])
        measured_robot_angle = robot_left_state[2]
        tag_offsets[self.robot_id] = true_robot_angle - measured_robot_angle

        if self.use_object:
            object_left_state, object_right_state = left_state[3:], right_state[3:]
            true_object_vector = (object_left_state - object_right_state)[:2]
            true_object_angle = np.arctan2(true_object_vector[1], true_object_vector[0])
            measured_object_angle = object_left_state[2]            
            tag_offsets[self.object_id] = true_object_angle - measured_object_angle

        np.save(self.tag_offset_path, tag_offsets)

    def train_model(self):
        idx = self.replay_buffer.capacity if self.replay_buffer.full else self.replay_buffer.idx
        states = self.replay_buffer.states[:idx-1]
        actions = self.replay_buffer.actions[:idx-1]
        next_states = self.replay_buffer.states[1:idx]

        training_losses, test_losses, discrim_training_losses, discrim_test_losses, test_idx = self.agent.train(
                states, actions, next_states, set_scalers=True, epochs=400, discrim_epochs=5, batch_size=1000, use_all_data=False)

        training_losses = np.array(training_losses).squeeze()
        test_losses = np.array(test_losses).squeeze()
        discrim_training_losses = np.array(discrim_training_losses).squeeze()
        discrim_test_losses = np.array(discrim_test_losses).squeeze()

        print("\nMIN TEST LOSS EPOCH:", test_losses.argmin())
        print("MIN TEST LOSS:", test_losses.min())

        print("\nMIN DISCRIM TEST LOSS EPOCH:", discrim_test_losses.argmin())
        print("MIN DISCRIM TEST LOSS:", discrim_test_losses.min())

        fig, axes = plt.subplots(1, 4)
        axes[0].plot(np.arange(len(training_losses)), training_losses, label="Training Loss")
        axes[1].plot(np.arange(-1, len(test_losses)-1), test_losses, label="Test Loss")
        axes[2].plot(np.arange(len(discrim_training_losses)), discrim_training_losses, label="Discriminator Training Loss")
        axes[3].plot(np.arange(len(discrim_test_losses)), discrim_test_losses, label="Discriminator Test Loss")

        axes[0].set_yscale('log')
        axes[1].set_yscale('log')

        axes[0].set_title('Training Loss')
        axes[1].set_title('Test Loss')
        axes[2].set_title('Discriminator Training Loss')
        axes[3].set_title('Discriminator Test Loss')

        for ax in axes:
            ax.grid()

        axes[0].set_ylabel('Loss')
        axes[1].set_xlabel('Epoch')
        fig.set_size_inches(15, 5)

        plt.show()

    def define_goal_trajectory(self):
        # self.front_left_corner = np.array([-0.1, -1.15])
        # self.back_right_corner = np.array([-1.9, 0.1])
        self.front_left_corner = np.array([-0.03, -1.4])
        self.back_right_corner = np.array([-2.5, 0.05])
        corner_range = self.back_right_corner - self.front_left_corner

        # max distance between robot and object(gives adequate buffer space near perimeter)
        max_sep_rel = abs(0.3/corner_range[0])
        back_circle_center_rel = np.array([(0.75*max_sep_rel) + 0.25, 0.5])
        front_circle_center_rel = np.array([0.75 - (0.75*max_sep_rel), 0.5])

        self.back_circle_center = back_circle_center_rel * corner_range + self.front_left_corner
        self.front_circle_center = front_circle_center_rel * corner_range + self.front_left_corner
        self.radius = np.linalg.norm(self.back_circle_center - self.front_circle_center) / 2

    def update_state(self, msg):
        if (self.halted or not self.initialized) and not self.calibrating:
            return

        found_robot, found_object = False, False
        for marker in msg.markers:
            if marker.id == self.robot_id:
                state = self.robot_state
                found_robot = True
            elif marker.id == self.object_id:
                state = self.object_state
                found_object = True
            else:
                continue

            o = marker.pose.pose.orientation
            o_list = [o.x, o.y, o.z, o.w]
            x, y, z = euler_from_quaternion(o_list)

            if abs(np.sin(x)) > self.flat_lim or abs(np.sin(y)) > self.flat_lim and self.started:
                print(f"{'ROBOT' if marker.id == self.robot_id else 'OBJECT'} MARKER NOT FLAT ENOUGH")
                print("sin(x):", np.sin(x), "|| sin(y)", np.sin(y))
                self.not_found = True
                return

            if hasattr(self, "tag_offsets"):
                z += self.tag_offsets[marker.id]

            state[0] = marker.pose.pose.position.x
            state[1] = marker.pose.pose.position.y
            state[2] = z % (2 * np.pi)

        if self.use_object:
            self.not_found = not (found_robot and found_object)
        else:
            self.not_found = not found_robot
        self.n_updates += 0 if self.not_found else 1

        if self.not_found:
            return

        if not self.calibrating:
            state_for_last_goal = self.robot_state if self.robot_goals else self.object_state
            last_goal = self.last_goal if self.started else state_for_last_goal
            dist_loss, heading_loss, perp_loss = self.agent.compute_losses(self.get_state(wait=False), last_goal, self.get_goal(), current=True, robot_goals=self.robot_goals)
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
        self.init_goal = self.get_goal(random=False)
        while not rospy.is_shutdown():
            if self.started:
                robot_state_to_plot = self.robot_state.copy()
                object_state_to_plot = self.object_state.copy()
                self.plot_robot_states.append(robot_state_to_plot)
                self.plot_object_states.append(object_state_to_plot)
                self.plot_goals.append(self.last_goal.copy())

                if self.plot:
                    self.plot_states_and_goals()
            elif self.dist_to_start() < self.tolerance:
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

    def plot_states_and_goals(self):
        plot_goals = np.array(self.plot_goals)
        plot_robot_states = np.array(self.plot_robot_states)
        plot_object_states = np.array(self.plot_object_states)
        plt.plot(plot_goals[:, 0] * -1, plot_goals[:, 1], color="green", linewidth=1.5, marker="*", label="Goal Trajectory")
        plt.plot(plot_robot_states[:, 0] * -1, plot_robot_states[:, 1], color="red", linewidth=1.5, marker=">", label="Robot Trajectory")
        plt.plot(plot_object_states[:, 0] * -1, plot_object_states[:, 1], color="blue", linewidth=1.5, marker=".", label="Object Trajectory")
        if self.first_plot:
            plt.legend()
            plt.ion()
            plt.show()
            self.first_plot = False
        plt.draw()
        plt.pause(0.0001)

    def dist_to_start(self):
        state = self.get_state(wait=False).squeeze()
        state = state[:3] if self.robot_goals else state[3:]
        return np.linalg.norm((state - self.init_goal)[:2])

    def step(self):
        if not self.pretrain and self.steps == self.random_steps:
            self.train_model()

        state = self.get_state()
        if state is None:
            print("MARKERS NOT VISIBLE")
            return False

        action = self.get_take_action(state)
        if action is None:
            print("MARKERS NOT VISIBLE")
            return False

        self.collect_training_data(state, action)
        self.update_model_online()
        self.check_rollout_finished()

        self.steps += 1
        return True

    def get_state(self, wait=True):
        if self.n_avg_states > 1 and not wait:
            robot_states, object_states = [], []
            while len(robot_states) < self.n_avg_states:
                if wait:
                    if self.n_updates == self.last_n_updates:
                        rospy.sleep(0.0001)
                    else:
                        robot_states.append(self.robot_state.copy())
                        object_states.append(self.object_state.copy())
                        self.last_n_updates = self.n_updates

            robot_state = np.array(robot_states).squeeze().mean(axis=0)
            object_state = np.array(object_states).squeeze().mean(axis=0)
        else:
            if wait:
                while self.n_updates == self.last_n_updates:
                    rospy.sleep(0.0001)
                self.last_n_updates = self.n_updates

            robot_state = self.robot_state.copy()
            object_state = self.object_state.copy()

        if self.use_object:
            current_state = np.concatenate((robot_state, object_state), axis=0)
        else:
            current_state = robot_state

        return current_state

    def get_take_action(self, state):
        goal = self.get_goal()
        state_for_last_goal = state[:3] if self.robot_goals else state[3:]
        last_goal = self.last_goal if self.started else state_for_last_goal

        if self.steps >= self.random_steps and np.random.rand() > self.epsilon:
            action = self.agent.mpc_action(state, last_goal, goal, self.action_range, n_steps=self.mpc_steps,
                                           n_samples=self.mpc_samples, perp_weight=self.perp_weight,
                                           heading_weight=self.heading_weight, dist_weight=self.dist_weight,
                                           norm_weight=self.norm_weight, sep_weight=self.sep_weight if self.started else 0.,
                                           discrim_weight=self.discrim_weight, heading_diff_weight=self.heading_diff_weight,
                                           dist_bonus_weight=self.dist_bonus_weight, robot_goals=self.robot_goals, mpc_softmax=self.mpc_softmax).detach().numpy()
        else:
            print("TAKING RANDOM ACTION")
            action = np.random.uniform(*self.action_range, size=(1, self.action_range.shape[-1])).squeeze()

        action = np.clip(action, *self.action_range)
        action_req = RobotCmd()
        action_req.left_pwm = action[0]
        action_req.right_pwm = action[1]
        action_req.duration = self.duration

        self.service_proxy(action_req, f"kami{self.robot_id}")
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
        print("ACTION:", action)
        print("ACTION NORM:", np.linalg.norm(action) / np.sqrt(2), "\n")
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
            if rospy.get_time() - time > 2:
                return None
            rospy.sleep(0.001)

        return action

    def get_goal(self, random=True):
        t_rel = (self.time_elapsed % self.lap_time) / self.lap_time
        if not self.started and random:
            t_rel += np.random.uniform(0, 0.01)

        if t_rel < 0.5:
            theta = t_rel * 2 * 2 * np.pi
            center = self.front_circle_center
        else:
            theta = np.pi - ((t_rel - 0.5) * 2 * 2 * np.pi)
            center = self.back_circle_center
        # if t_rel < 0.25:
        #     theta = 2 * np.pi * t_rel / 0.5
        #     center = self.back_circle_center
        # elif 0.25 <= t_rel < 0.75:
        #     theta = -2 * np.pi * (t_rel - 0.25) / 0.5
        #     center = self.front_circle_center
        # else:
        #     theta = 2 * np.pi * (t_rel - 0.5) / 0.5
        #     center = self.back_circle_center

        # theta -= np.pi / 4
        goal = center + np.array([np.cos(theta), np.sin(theta)]) * self.radius
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
            self.plot_states_and_goals()
            plt.savefig("/home/bvanbuskirk/Desktop/lap_plots/Transitions:{}_robot:{}".format(self.replay_buffer.idx, self.robot_goals))
            plt.pause(1.0)
            plt.close()

        if self.laps == self.n_rollouts:
            self.dump_performance_metrics()
            self.done = True

    def dump_performance_metrics(self):
        path = "/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/sim/data/loss/"
        np.save(path + f"losses", self.losses)

        dist_losses, heading_losses, perp_losses, total_losses = self.losses.T
        data = np.array([[dist_losses.mean(), dist_losses.std(), dist_losses.min(), dist_losses.max()],
                         [perp_losses.mean(), perp_losses.std(), perp_losses.min(), perp_losses.max()],
                         [heading_losses.mean(), heading_losses.std(), heading_losses.min(), heading_losses.max()],
                         [total_losses.mean(), total_losses.std(), total_losses.min(), total_losses.max()]])

        print("rows: (dist, perp, heading, total)")
        print("cols: (mean, std, min, max)")
        print("DATA:", data, "\n")

    def collect_training_data(self, state, action):
        self.replay_buffer.add(state, action)

        if self.replay_buffer.idx % self.save_freq == 0:
            print(f"\nSAVING REPLAY BUFFER WITH {self.replay_buffer.capacity if self.replay_buffer.full else self.replay_buffer.idx} TRANSITIONS\n")
            with open("/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/replay_buffers/buffer.pkl", "wb") as f:
                pkl.dump(self.replay_buffer, f)

    def update_model_online(self):
        return
        if (self.replay_buffer.full or self.replay_buffer.idx > 50) and (self.pretrain or (not self.pretrain and self.steps >= self.random_steps)):
            for model in self.agent.models:
                for _ in range(self.gradient_steps):
                    states, actions, next_states = self.replay_buffer.sample(200)
                    model.update(states, actions, next_states)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do MPC.')
    parser.add_argument('-robot_id', type=int, help='robot id for rollout')
    parser.add_argument('-object_id', type=int, default=-1, help='object id for rollout')
    parser.add_argument('-mpc_steps', type=int)
    parser.add_argument('-mpc_samples', type=int)
    parser.add_argument('-n_rollouts', type=int)
    parser.add_argument('-tolerance', type=float, default=0.05)
    parser.add_argument('-lap_time', type=float)
    parser.add_argument('-calibrate', action='store_true')
    parser.add_argument('-plot', action='store_true')
    parser.add_argument('-new_buffer', action='store_true')
    parser.add_argument('-pretrain', action='store_true')
    parser.add_argument('-robot_goals', action='store_true')
    parser.add_argument('-scale', action='store_true')
    parser.add_argument('-mpc_softmax', action='store_true')

    args = parser.parse_args()

    r = RealMPC(args.robot_id, args.object_id, args.mpc_steps, args.mpc_samples, args.n_rollouts, args.tolerance,
                args.lap_time, args.calibrate, args.plot, args.new_buffer, args.pretrain, args.robot_goals, args.scale, args.mpc_softmax)
    r.run()
