#!/usr/bin/python3

import os
import numpy as np

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from ros_stuff.msg import MultiRobotCmd
import trajectories

from utils import build_action_msg, YAW_OFFSET_PATH


class Environment:
    def __init__(self, robot_pos_dict, robot_vel_dict, object_pos, object_vel, corner_pos, action_receipt_dict, params, calibrate=False, precollecting=False):
        self.params = params

        self.post_action_sleep_time = 0.5
        self.action_duration = 0.4 if self.use_object else 0.3

        self.episode_step = 0
        self.reverse_episode = True
        self.original_episode_length = self.episode_length

        self.robot_pos_dict = robot_pos_dict
        self.robot_vel_dict = robot_vel_dict
        self.object_pos = object_pos
        self.object_vel = object_vel
        self.corner_pos = corner_pos
        self.action_receipt_dict = action_receipt_dict

        self.episode_states = []
        self.camera_imgs = []

        self.cv_bridge = CvBridge()
        self.action_publisher = rospy.Publisher("/action_topic", MultiRobotCmd, queue_size=1)

        if not calibrate and not precollecting:
            self.camera_img_subscriber = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback, queue_size=1)
            rospy.sleep(0.5)        # wait for states to be published and set

            if self.trajectory == "S":
                self.goal_fn = trajectories.S_trajectory
                self.goals = self.goal_fn(self.episode_length, self.mpc_horizon)
            elif self.trajectory == "W":
                self.goal_fn = trajectories.W_trajectory
                self.goals, self.episode_length = self.goal_fn(self.episode_length, self.mpc_horizon)
            elif self.trajectory == "8":
                self.goal_fn = trajectories.figure8_trajectory
                self.goals = self.goal_fn(self.episode_length, self.mpc_horizon)
            else:
                raise ValueError

            self.goals = np.concatenate((self.goals, np.zeros((len(self.goals), 1))), axis=1)

        if not calibrate:
            assert os.path.exists(YAW_OFFSET_PATH)
            self.yaw_offsets = np.load(YAW_OFFSET_PATH)
        else:
            self.yaw_offsets = np.zeros(10)

        self.corner_pos_perm = corner_pos

        self.save_states = False

    def __getattr__(self, key):
        return self.params[key]

    def image_callback(self, msg):
        cv_img = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        self.current_image = cv_img

        if self.save_real_video:
            self.camera_imgs.append(cv_img)

    def step(self, action, reset=False):
        action_msg = build_action_msg(action, self.action_duration, self.robot_ids)
        self.action_publisher.publish(action_msg)

        while not all([receipt for receipt in self.action_receipt_dict.values()]):
            rospy.sleep(0.01)

        rospy.sleep(self.action_duration + self.post_action_sleep_time)
        for id in self.action_receipt_dict:
            self.action_receipt_dict[id] = False

        state = self.get_state()
        self.episode_states.append(state)

        if not reset:
            self.episode_step += 1

        done = (self.episode_step == self.episode_length)

        return state, done

    def reset(self, agent, replay_buffer):
        self.episode_step = 0
        self.reverse_episode = not self.reverse_episode

        if self.trajectory == "W":
            self.goals, _ = self.goal_fn(self.original_episode_length, self.mpc_horizon, reverse=self.reverse_episode)
        else:
            self.goals = self.goal_fn(self.episode_length, self.mpc_horizon, reverse=self.reverse_episode)
        self.goals = np.concatenate((self.goals, np.zeros((len(self.goals), 1))), axis=1)

        init_goal = self.get_next_n_goals(1)
        state = self.get_state()

        if self.robot_goals:
            reference_state = state[:3]
        else:
            reference_state = state[-3:]

        reset_step = 0

        while np.linalg.norm((init_goal - reference_state)[:, :2]) > self.tolerance:
            if replay_buffer.size >= self.eval_buffer_size:
                self.update_online = False

            action, predicted_next_state = agent.get_action(state, init_goal)
            next_state, _ = self.step(action, reset=True)

            if next_state is not None:
                replay_buffer.add(state, action, next_state)
                state = next_state
            else:
                state = self.get_state()

            if self.robot_goals:
                reference_state = state[:3]
            else:
                reference_state = state[-3:]

            if self.update_online:
                for model in agent.models:
                    for _ in range(self.utd_ratio):
                        model.update(*replay_buffer.sample(self.batch_size))

            print("\nRESET STEP:", reset_step)
            print("DISTANCE:", np.linalg.norm((init_goal - reference_state)[:, :2]))
            reset_step += 1

        self.episode_states = []
        self.camera_imgs = []

        return state

    def render(self, done=False, episode=0):
        real_img = self.current_image.copy()

        states = np.array(self.episode_states)
        robot_states_dict = {id: states[:, 3*i:3*(i+1)] for i, id in enumerate(self.robot_ids)}
        object_states = states[:, -3:]
        goal_states = self.get_next_n_goals(self.episode_step, episode_step=0)

        fig, ax = plt.subplots()
        linewidth = 0.7

        ax.plot(goal_states[:, 0], goal_states[:, 1], color="green", linewidth=linewidth, marker="*", label="Goal Trajectory")
        if self.save_states and done:
            np.save(f"/home/arovinsky/mar1_dump/goal_states_ep{episode}.npy", goal_states)

        colors = ["blue", "purple", "orange"]
        for i, (id, robot_states) in enumerate(robot_states_dict.items()):
            ax.plot(robot_states[:, 0], robot_states[:, 1], color=colors[i], linewidth=linewidth,
                    marker=">", markersize=5, label=f"Robot{id} Trajectory")
            if self.save_states and done:
                np.save(f"/home/arovinsky/mar1_dump/robot_states_{id}_ep{episode}.npy", robot_states)

        if self.use_object:
            ax.plot(object_states[:, 0], object_states[:, 1], color="red", linewidth=linewidth,
                    marker=".", markersize=9, label="Object Trajectory")
            if self.save_states and done:
                np.save(f"/home/arovinsky/mar1_dump/object_states_ep{episode}.npy", object_states)

        ax.axis('equal')
        ax.set_xlim((0, self.corner_pos_perm[0]))
        ax.set_ylim((0, self.corner_pos_perm[1]))
        ax.legend()
        ax.set_title('Position of System Elements', fontweight='bold')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')

        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,)).copy()
        plot_img[..., [2, 0]] = plot_img[..., [0, 2]]

        plt.close(fig)

        return plot_img, real_img

    def get_state(self):
        states = []

        for id in self.robot_pos_dict:
            states.append(self.get_state_from_id(id))

        if self.use_object:
            states.append(self.get_state_from_id(self.object_id))

        out_of_bounds = lambda pos: np.any(pos[:2] > self.corner_pos[:2]) or np.any(pos[:2] < 0)

        valid = True
        for state in states:
            valid = valid and not out_of_bounds(state)

        if not valid:
            print("\nOUT OF BOUNDS\n")
            import pdb;pdb.set_trace()
            state = None
        else:
            state = np.concatenate(states, axis=0)

        return state

    def get_state_from_id(self, id):
        if id == self.object_id:
            pos = self.object_pos.copy()
        else:
            pos = self.robot_pos_dict[id].copy()

        pos[2] = (pos[2] + self.yaw_offsets[id]) % (2 * np.pi)
        return pos

    def get_next_n_goals(self, n, episode_step=None):
        if episode_step is None:
            episode_step = self.episode_step

        start_step = episode_step
        end_step = start_step + n
        assert start_step < end_step

        goals = self.goals[start_step:end_step]

        return goals
