#!/usr/bin/python3

import os
import argparse
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import rospy

from real_mpc_dynamics import *
from replay_buffer import ReplayBuffer
from state_subscriber import StateSubscriber

from ros_stuff.msg import RobotCmd, ProcessedStates
from ros_stuff.srv import CommandAction

from ar_track_alvar_msgs.msg import AlvarMarkers
from tf.transformations import euler_from_quaternion

# seed for reproducibility
SEED = 0
import torch; torch.manual_seed(SEED)
np.random.seed(SEED)


class RealMPC():
    def __init__(self, robot_id, object_id, mpc_steps, mpc_samples, n_rollouts, tolerance, lap_time, calibrate, plot,
                 new_buffer, pretrain, robot_goals, scale, mpc_softmax, save_freq, online, mpc_refine_iters, pretrain_samples):
        # flags for different stages of eval
        self.started = False
        self.done = False

        # counters
        self.steps = 0
        self.lag_states = 0

        # AR tag ids for state lookup
        self.robot_id = robot_id
        self.object_id = object_id

        # states
        self.robot_state = np.zeros(3)
        self.object_state = np.zeros(3)
        self.corner_state = np.zeros(3)

        # MPC params
        self.mpc_steps = mpc_steps
        self.mpc_samples = mpc_samples
        self.mpc_softmax = mpc_softmax
        self.mpc_refine_iters = mpc_refine_iters
        self.use_object = (self.object_id >= 0)

        # action params
        max_pwm = 0.999
        self.action_range = np.array([[-max_pwm, -max_pwm], [max_pwm, max_pwm]])
        self.duration = 0.2

        # online data collection/learning params
        self.random_steps = 0 if pretrain else 500
        self.gradient_steps = 2
        self.online = online
        self.save_freq = save_freq
        self.pretrain_samples = pretrain_samples

        # system params
        self.n_avg_states = 1
        self.n_wait_updates = 1
        self.n_clip = 3
        self.max_lag_states = 3

        # misc
        self.n_rollouts = n_rollouts
        self.tolerance = tolerance
        self.lap_time = lap_time
        self.plot = plot
        self.robot_goals = robot_goals
        self.scale = scale
        self.pretrain = pretrain

        if os.path.exists("/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/replay_buffers/buffer.pkl") and not new_buffer:
            with open("/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/replay_buffers/buffer.pkl", "rb") as f:
                self.replay_buffer = pkl.load(f)
        else:
            state_dim = 6 if self.use_object else 3
            self.replay_buffer = ReplayBuffer(capacity=10000, state_dim=state_dim, action_dim=2)

        rospy.init_node("laptop_client_mpc")

        print("waiting for /processed_state topic from state publisher")
        rospy.Subscriber("/processed_state", ProcessedStates, self.update_state, queue_size=1)
        print("subscribed to /processed_state")

        robot_id = self.robot_id if False else 1
        print(f"waiting for robot {robot_id} service")
        rospy.wait_for_service(f"/kami{robot_id}/server")
        self.service_proxy = rospy.ServiceProxy(f"/kami{robot_id}/server", CommandAction)
        print("connected to robot service")

        self.yaw_offset_path = "/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/sim/data/yaw_offsets.npy"
        if not os.path.exists(self.yaw_offset_path) or calibrate:
            self.calibrate()

        self.yaw_offsets = np.load(self.yaw_offset_path)

        self.define_goal_trajectory()

        # weights for MPC cost terms
        # self.perp_weight = 4.
        # self.heading_weight = 0.8
        # self.dist_weight = 3.0
        # self.norm_weight = 0.0
        # self.dist_bonus_weight = 10.
        # self.sep_weight = 0.0
        # self.discrim_weight = 0.
        # self.heading_diff_weight = 0.0

        self.perp_weight = 0.
        self.heading_weight = 0.0
        self.dist_weight = 1.
        self.norm_weight = 0.0
        self.dist_bonus_weight = 0.
        self.sep_weight = 0.
        self.discrim_weight = 0.0
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

        self.agent = MPCAgent(seed=SEED, dist=True, scale=self.scale, hidden_dim=250, hidden_depth=2,
                              lr=0.001, dropout=0.0, ensemble=1, use_object=self.use_object)

        if pretrain:
            self.train_model()

        # for g in self.agent.models[0].optimizer.param_groups:
        #     g['lr'] = 3e-4

        np.set_printoptions(suppress=True)

    def calibrate(self):
        yaw_offsets = np.zeros(10)

        input(f"Place robot/object on the left calibration point, aligned with the calibration line and hit enter.")
        left_state = self.get_state()
        input(f"Place robot/object on the right calibration point, aligned with the calibration line and hit enter.")
        right_state = self.get_state()

        robot_left_state, robot_right_state = left_state[:3], right_state[:3]
        true_robot_vector = (robot_left_state - robot_right_state)[:2]
        true_robot_angle = np.arctan2(true_robot_vector[1], true_robot_vector[0])
        measured_robot_angle = robot_left_state[2]
        yaw_offsets[self.robot_id] = true_robot_angle - measured_robot_angle

        if self.use_object:
            object_left_state, object_right_state = left_state[3:], right_state[3:]
            true_object_vector = (object_left_state - object_right_state)[:2]
            true_object_angle = np.arctan2(true_object_vector[1], true_object_vector[0])
            measured_object_angle = object_left_state[2]
            yaw_offsets[self.object_id] = true_object_angle - measured_object_angle

        np.save(self.yaw_offset_path, yaw_offsets)

    def train_model(self):
        idx = self.replay_buffer.capacity if self.replay_buffer.full else self.replay_buffer.idx

        # states = self.replay_buffer.states[:idx-1]
        # actions = self.replay_buffer.actions[:idx-1]
        # next_states = self.replay_buffer.states[1:idx]

        # random_idx = np.random.permutation(len(states))[:self.pretrain_samples]
        # states = states[random_idx]
        # actions = actions[random_idx]
        # next_states = next_states[random_idx]

        states, actions, next_states = self.replay_buffer.sample(self.pretrain_samples)

        training_losses, test_losses, discrim_training_losses, discrim_test_losses, test_idx = self.agent.train(
                states, actions, next_states, set_scalers=True, epochs=2100, discrim_epochs=5, batch_size=1000, use_all_data=False)

        training_losses = np.array(training_losses).squeeze()
        test_losses = np.array(test_losses).squeeze()
        discrim_training_losses = np.array(discrim_training_losses).squeeze()
        discrim_test_losses = np.array(discrim_test_losses).squeeze()

        print("\nMIN TEST LOSS EPOCH:", test_losses.argmin())
        print("MIN TEST LOSS:", test_losses.min())

        print("\nMIN DISCRIM TEST LOSS EPOCH:", discrim_test_losses.argmin())
        print("MIN DISCRIM TEST LOSS:", discrim_test_losses.min())

        state_delta = self.agent.dtu.state_delta_xysc(states, next_states)

        test_state, test_action = states[test_idx], actions[test_idx]
        test_state_delta = dtu.dcn(state_delta[test_idx])

        pred_state_delta = dtu.dcn(self.agent.models[0](test_state, test_action, sample=False))
        # pred_state_delta = agent.get_prediction(test_states, test_actions, sample=False, scale=args.scale, delta=True, use_ensemble=False)

        error = abs(pred_state_delta - test_state_delta)
        print("\nERROR MEAN:", error.mean(axis=0))
        print("ERROR STD:", error.std(axis=0))
        print("ERROR MAX:", error.max(axis=0))
        print("ERROR MIN:", error.min(axis=0))

        diffs = abs(test_state_delta)
        print("\nACTUAL MEAN:", diffs.mean(axis=0))
        print("ACTUAL STD:", diffs.std(axis=0))

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
        # # self.front_left_corner = np.array([-0.1, -1.15])
        # # self.back_right_corner = np.array([-1.9, 0.1])
        # self.front_left_corner = np.array([-0.03, -1.4])
        # self.back_right_corner = np.array([-2.5, 0.05])
        # corner_range = self.back_right_corner - self.front_left_corner
        corner_range = self.corner_state

        # max distance between robot and object(gives adequate buffer space near perimeter)
        max_sep_rel = abs(0.3/corner_range[0])
        back_circle_center_rel = np.array([(0.75*max_sep_rel) + 0.25, 0.5])
        front_circle_center_rel = np.array([0.75 - (0.75*max_sep_rel), 0.5])

        self.back_circle_center = back_circle_center_rel * corner_range + self.front_left_corner
        self.front_circle_center = front_circle_center_rel * corner_range + self.front_left_corner
        self.radius = np.linalg.norm(self.back_circle_center - self.front_circle_center) / 2

    def update_state(self, msg):
        rs, os, cs = msg.robot_state, msg.object_state, msg.corner_state
        self.robot_state = np.array([rs.x, rs.y, rs.yaw])
        self.object_state = np.array([os.x, os.y, os.yaw])
        self.corner_state = np.array([cs.x, cs.y, cs.yaw])

    def record_losses(self):
        if self.started:
            dist_loss, heading_loss, perp_loss = self.agent.compute_losses(
                self.get_state(), self.last_goal, self.get_goal(),
                current=True, robot_goals=self.robot_goals
                )
            total_loss = dist_loss * (self.dist_weight + heading_loss * self.heading_weight + perp_loss * self.perp_weight)

            self.stamped_losses[1:] = self.stamped_losses[:-1]
            self.stamped_losses[0] = [rospy.get_time()] + [i.detach().numpy() for i in [dist_loss, heading_loss, perp_loss, total_loss]]

    def record_plot_states(self):
        robot_state_to_plot = self.robot_state.copy()
        object_state_to_plot = self.object_state.copy()
        self.plot_robot_states.append(robot_state_to_plot)
        self.plot_object_states.append(object_state_to_plot)
        self.plot_goals.append(self.last_goal.copy())

        if self.plot:
            self.plot_states_and_goals()

    def run(self):
        rospy.sleep(0.2)
        while self.state_subscriber.n_full_updates < 3 and not rospy.is_shutdown():
            print("WAITING FOR STATE SUBSCRIBER TO UPDATE")
            rospy.sleep(0.2)

        self.first_plot = True
        self.init_goal = self.get_goal(random=False)
        while not rospy.is_shutdown():
            if self.started:
                self.record_plot_states()
            elif self.dist_to_start() < self.tolerance and (self.pretrain or self.steps >= self.random_steps):
                self.started = True
                self.last_goal = self.get_goal()
                self.time_elapsed = self.duration / 2

            self.step()
            self.record_losses()

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
        state = self.get_state().squeeze()
        state = state[:3] if self.robot_goals else state[3:]
        return np.linalg.norm((state - self.init_goal)[:2])

    def step(self):
        if not self.pretrain and self.steps == self.random_steps:
            self.train_model()

        state = self.get_state()
        action = self.get_take_action(state)
        terminal = False

        if self.n_full_updates == self.state_subscriber.n_full_updates:
            self.lag_states += 1
            if self.lag_states > self.max_lag_states:
                print("\nHALTED DUE TO TRACKING LAG\n")
                import pdb;pdb.set_trace()
                terminal = True
        else:
            self.lag_states = 0

        self.collect_training_data(state, action, terminal=terminal)

        if self.online:
            self.update_model_online()

        self.check_rollout_finished()

        self.steps += 1
        return True

    def get_state(self):
        if self.use_object:
            return np.concatenate((self.robot_state, self.object_state), axis=0)
        else:
            return self.robot_state

    def get_take_action(self, state):
        goal = self.get_goal()
        state_for_last_goal = state[:3] if self.robot_goals else state[3:]
        last_goal = self.last_goal if self.started else state_for_last_goal

        if self.steps >= self.random_steps:
            action = self.agent.mpc_action(state, last_goal, goal, self.action_range, n_steps=self.mpc_steps,
                                           n_samples=self.mpc_samples, perp_weight=self.perp_weight,
                                           heading_weight=self.heading_weight, dist_weight=self.dist_weight,
                                           norm_weight=self.norm_weight, sep_weight=self.sep_weight if self.started else 0.,
                                           discrim_weight=self.discrim_weight, heading_diff_weight=self.heading_diff_weight,
                                           dist_bonus_weight=self.dist_bonus_weight, robot_goals=self.robot_goals,
                                           mpc_softmax=self.mpc_softmax, mpc_refine_iters=self.mpc_refine_iters)
            self.time_elapsed += self.duration if self.started else 0
        else:
            print("TAKING RANDOM ACTION")
            action = np.random.uniform(*self.action_range, size=(1, self.action_range.shape[-1])).squeeze()

        action = np.clip(action, *self.action_range)
        action_req = RobotCmd()
        action_req.left_pwm = action[0]
        action_req.right_pwm = action[1]
        action_req.duration = self.duration

        # record n full updates just before taking action to check action was tracked
        self.n_full_updates = self.state_subscriber.n_full_updates
        self.service_proxy(action_req, f"kami{self.robot_id}")

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
        # n_updates = self.n_updates
        # time = rospy.get_time()
        # while self.n_updates - n_updates < self.n_wait_updates:
        #     if rospy.get_time() - time > 2:
        #         return None
        #     rospy.sleep(0.001)

        return action

    def get_goal(self, random=True):
        t_rel = (self.time_elapsed % self.lap_time) / self.lap_time
        # if not self.started and random:
        #     t_rel += np.random.uniform(0, 0.01)

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
    parser.add_argument('-online', action='store_true')
    parser.add_argument('-save_freq', type=int, default=50)
    parser.add_argument('-mpc_refine_iters', type=int, default=1)
    parser.add_argument('-pretrain_samples', type=int, default=500)

    args = parser.parse_args()

    r = RealMPC(args.robot_id, args.object_id, args.mpc_steps, args.mpc_samples, args.n_rollouts, args.tolerance,
                args.lap_time, args.calibrate, args.plot, args.new_buffer, args.pretrain, args.robot_goals, args.scale,
                args.mpc_softmax, args.save_freq, args.online, args.mpc_refine_iters, args.pretrain_samples)
    r.run()
