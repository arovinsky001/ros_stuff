#!/usr/bin/python3

import os
import numpy as np
# import gym
# from gym import register

import rospy
from std_msgs.msg import Time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from ros_stuff.msg import MultiRobotCmd

from utils import build_action_msg, YAW_OFFSET_PATH

cv_bridge = CvBridge()



class Environment:
    def __init__(self, robot_pos_dict, robot_vel_dict, object_pos, object_vel, corner_pos, action_receipt_dict, params, agent, calibrate=False):
        self.params = params
        self.agent = agent

        self.init_tolerance = 0.04
        self.post_action_sleep_time = 0.3
        self.action_duration = 0.2

        self.episode_step = 0
        self.begin_episode_step = np.floor(np.random.rand() * self.episode_length)
        self.last_action_timestamp = 0
        self.reverse_episode = False

        self.robot_pos_dict = robot_pos_dict
        self.robot_vel_dict = robot_vel_dict
        self.object_pos = object_pos
        self.object_vel = object_vel
        self.corner_pos = corner_pos
        self.action_receipt_dict = action_receipt_dict

        self.action_publisher = rospy.Publisher("/action_topic", MultiRobotCmd, queue_size=1)
        # self.camera_img_subscriber = rospy.Subscriber("/usb_cam/image_raw", Image, image_callback, queue_size=1)

        if not calibrate:
            # define centers and radius for figure-8 trajectory
            rospy.sleep(0.2)        # wait for states to be published and set

            if self.robot_goals:
                back_circle_center_rel = np.array([0.7, 0.5])
                front_circle_center_rel = np.array([0.4, 0.5])
            else:
                back_circle_center_rel = np.array([0.65, 0.5])
                front_circle_center_rel = np.array([0.4, 0.5])

            corner_pos = self.corner_pos.copy()
            self.back_circle_center = back_circle_center_rel * corner_pos[:2]
            self.front_circle_center = front_circle_center_rel * corner_pos[:2]
            self.radius = np.linalg.norm(self.back_circle_center - self.front_circle_center) / 2

            assert os.path.exists(YAW_OFFSET_PATH)
            self.yaw_offsets = np.load(YAW_OFFSET_PATH)
        else:
            self.yaw_offsets = np.zeros(10)

    def __getattr__(self, key):
        return self.params[key]

    def image_callback(self, msg):
        cv_img = cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        self.current_image = cv_img


    def step(self, action, reset=False):
        action_msg = build_action_msg(action, self.action_duration, self.robot_ids)
        self.action_publisher.publish(action_msg)

        while not all([receipt for receipt in self.action_receipt_dict.values()]):
            rospy.sleep(0.01)

        rospy.sleep(self.action_duration + self.post_action_sleep_time)
        for id in self.action_receipt_dict:
            self.action_receipt_dict[id] = False

        state = self.get_state()

        if not reset:
            self.episode_step += 1

        done = (self.episode_step == self.episode_length)

        return state, done

    def reset(self, replay_buffer):
        self.episode_step = 0
        self.begin_episode_step = np.floor(np.random.rand() * self.episode_length)

        init_goal = self.get_next_n_goals(1)
        state = self.get_state()

        while np.linalg.norm(init_goal - state) > self.init_tolerance:
            action = self.agent.get_action(state, init_goal)
            next_state, _ = self.step(action, reset=True)

            if state is not None and next_state is not None:
                replay_buffer.add(state, action, next_state)

            state = next_state

        self.reverse_episode = not self.reverse_episode

        return state
    

    def render(self):
        # get current state
        # make plot based on relevant states and goals (track self.states or something)
        # get real camera image from a ros subscriber

        real_img = self.current_image.copy()

        states = np.array(self.states)
        goals = np.array(self.goals)

        fig, ax = plt.subplots()

        ax.plot(goals[:, 0], goals[:, 1], color="green", linewidth=1.0, marker="*", label="Goal Trajectory")
        ax.plot(states[:, 0], states[:, 1], color="red", linewidth=1.0, marker=">", label="Robot Trajectory")

        if self.use_object:
            ax.plot(states[:, -3], states[:, -2], color="blue", linewidth=1.0, marker=".", label="Object Trajectory")

        ax.axis('equal')
        ax.set_xlim((self.corner[0], 0))
        ax.set_ylim((self.corner[1], 0))
        ax.legend()
        #fig.show()
        # Still giving type issues, but need to try running it
        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
         
        #  Don't think we need this but commenting out for now
        #  if len(self.goal_states) == 1 or not self.plot:
        #      ax = plt.gca()
        #      ax.axis('equal')
        #      plt.xlim((self.corner[0], 0))
        #      plt.ylim((self.corner[1], 0))
        #      plt.legend()
        #      plt.ion()
        #      plt.show()

        #  if save:
        #      plt.plot(start_state[0], start_state[1], color="orange", marker="D", label="Starting Point", markersize=6)
        #      plt.title(f"{'Reversed' if reverse_lap else 'Standard'} Trajectory")
        #      plt.xlabel("X Position")
        #      plt.ylabel("Y Position")
        #      plt.legend()
        #      plt.draw()
        #      plt.pause(1.)

        #      plt.savefig(self.plot_path + f"lap{laps}_rb{replay_buffer.size}.png")

        return plot_img, real_img

    def get_state(self):
        robot_pos_dict = self.robot_pos_dict.copy()
        object_pos = self.object_pos.copy()
        corner_pos = self.corner_pos.copy()

        states = []

        for id, robot_pos in robot_pos_dict.items():
            robot_pos[2] = (robot_pos[2] + self.yaw_offsets[id]) % (2 * np.pi)
            states.append(robot_pos)

        if self.use_object:
            object_pos[2] = (object_pos[2] + self.yaw_offsets[self.object_id]) % (2 * np.pi)
            states.append(object_pos)

        out_of_bounds = lambda pos: np.any(pos[:2] > corner_pos[:2]) or np.any(pos[:2] < 0)

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

    def get_goal(self, step_override=None):
        if step_override is not None:
            episode_step = step_override
        else:
            episode_step = self.episode_step

        t_rel = ((episode_step + self.begin_episode_step) % self.episode_length) / self.episode_length

        if t_rel < 0.5:
            theta = t_rel * 2 * 2 * np.pi
            center = self.front_circle_center
        else:
            theta = np.pi - ((t_rel - 0.5) * 2 * 2 * np.pi)
            center = self.back_circle_center

        goal = center + np.array([np.cos(theta), np.sin(theta)]) * self.radius
        return np.block([goal, 0.])

    def get_next_n_goals(self, n):
        goals = np.empty((n, 3))

        for i in range(n):
            step = self.episode_step + i * (-1 if self.reverse_episode else 1)
            goals[i] = self.get_goal(step_override=step)

        return goals


# register("kamigami-hardware-v0", Environment)
