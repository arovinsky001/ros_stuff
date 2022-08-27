#!/usr/bin/python3

import argparse
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import rospy

from real_mpc_dynamics import *
from kamigami_interface import KamigamiInterface
from ros_stuff.msg import RobotCmd

from tf.transformations import euler_from_quaternion

SAVE_PATH = "/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/sim/data/real_data_online400.npz"

class RealMPC(KamigamiInterface):
    def __init__(self, robot_id, object_id, mpc_steps, mpc_samples, n_rollouts, tolerance, lap_time, calibrate, plot,
                 new_buffer, pretrain, robot_goals, scale):
        self.halted = False
        self.initialized = False
        self.epsilon = 0.0
        self.steps = 0
        self.gradient_steps = 5
        self.random_steps = 0 if pretrain else 500

        super().__init__(robot_id, object_id, SAVE_PATH, calibrate, new_buffer=new_buffer)
        self.define_goal_trajectory()

        self.mpc_steps = mpc_steps
        self.mpc_samples = mpc_samples
        self.n_rollouts = n_rollouts
        self.tolerance = tolerance
        self.lap_time = lap_time
        self.plot = plot
        self.robot_goals = robot_goals
        self.scale = scale

        # weights for MPC cost terms
        self.perp_weight = 4.
        self.heading_weight = 0.8
        self.dist_weight = 3.0
        self.norm_weight = 0.0
        self.dist_bonus_factor = 10.
        self.sep_weight = 0.0
        self.discrim_weight = 0.
        self.heading_diff_weight = 0.0

        # self.perp_weight = 0.
        # self.heading_weight = 0.
        # self.dist_weight = 1.
        # self.norm_weight = 0.0
        # self.dist_bonus_factor = 0.
        # self.sep_weight = 0.
        # self.discrim_weight = 0.
        # self.heading_diff_weight = 0.0

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

        self.agent = MPCAgent(seed=0, dist=True, scale=self.scale, hidden_dim=200, hidden_depth=2,
                              lr=0.001, dropout=0.2, ensemble=1, use_object=True)

        if pretrain:
            idx = self.replay_buffer.capacity if self.replay_buffer.full else self.replay_buffer.idx
            states = self.replay_buffer.states[:idx-1]
            actions = self.replay_buffer.actions[:idx-1]
            next_states = self.replay_buffer.states[1:idx]

            training_losses, test_losses, discrim_training_losses, discrim_test_losses, test_idx = self.agent.train(
                    states, actions, next_states, set_scalers=True, epochs=400, discrim_epochs=5, batch_size=1000, use_all_data=False)

            training_losses = np.array(training_losses).squeeze()
            test_losses = np.array(test_losses).squeeze()
            discrim_training_losses = np.array(discrim_training_losses).squeeze()
            discrim_test_losses = np.array(discrim_training_losses).squeeze()

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

        for g in self.agent.models[0].optimizer.param_groups:
            g['lr'] = 3e-4

        np.set_printoptions(suppress=True)
        self.initialized = True

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

        super().update_state(msg)
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
                # plot robot and object
                robot_state_to_plot = self.robot_state.copy()
                object_state_to_plot = self.object_state.copy()
                self.plot_robot_states.append(robot_state_to_plot)
                self.plot_object_states.append(object_state_to_plot)  
                self.plot_goals.append(self.last_goal.copy())
                if self.plot:
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
        state = self.get_state(wait=False).squeeze()
        state = state[:3] if self.robot_goals else state[3:]
        return np.linalg.norm((state - self.init_goal)[:2])

    def step(self):
        state = self.get_state()
        if state is None:
            print("MARKERS NOT VISIBLE")
            return False

        action = self.get_take_actions(state)
        if action is None:
            print("MARKERS NOT VISIBLE")
            return False

        self.collect_training_data(state, action)
        self.update_model_online()
        self.check_rollout_finished()

        self.steps += 1
        return True

    def get_take_actions(self, state):
        goal = self.get_goal()
        state_for_last_goal = state[:3] if self.robot_goals else state[3:]
        last_goal = self.last_goal if self.started else state_for_last_goal

        if np.random.rand() > self.epsilon and self.steps > self.random_steps:
            action = self.agent.mpc_action(state, last_goal, goal, self.action_range, n_steps=self.mpc_steps,
                                           n_samples=self.mpc_samples, perp_weight=self.perp_weight,
                                           heading_weight=self.heading_weight, dist_weight=self.dist_weight,
                                           norm_weight=self.norm_weight, sep_weight=self.sep_weight if self.started else 0.,
                                           discrim_weight=self.discrim_weight, heading_diff_weight=self.heading_diff_weight,
                                           robot_goals=self.robot_goals).detach().numpy()
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
        if self.replay_buffer.full or self.replay_buffer.idx > 50:
            for model in self.agent.models:
                for _ in range(self.gradient_steps):
                    states, actions, next_states = self.replay_buffer.sample(200)
                    model.update(states, actions, next_states)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do MPC.')
    parser.add_argument('-robot_id', type=int, help='robot id for rollout')
    parser.add_argument('-object_id', type=int, help='object id for rollout')
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

    args = parser.parse_args()

    r = RealMPC(args.robot_id, args.object_id, args.mpc_steps, args.mpc_samples, args.n_rollouts, args.tolerance,
                args.lap_time, args.calibrate, args.plot, args.new_buffer, args.pretrain, args.robot_goals, args.scale)
    r.run()
