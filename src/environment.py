#!/usr/bin/python3

import os
import numpy as np
# import gym
# from gym import register

import rospy
from std_msgs.msg import Time
from ros_stuff.msg import RobotCmd

from utils import build_action_msg
from calibrate import YAW_OFFSET_PATH


# class Environment(gym.Env):
    # def __init__(self,
    #     agent: MPCAgent,
    #     n_robots: int = 1,
    #     action_duration: float = 0.2,
    #     episode_length: int = 150,
    # ):

class Environment:
    def __init__(self, robot_pos, object_pos, corner_pos, robot_vel, object_vel, action_timestamp, params, agent):
        self.params = params
        self.agent = agent

        self.init_tolerance = 0.04
        self.post_action_sleep_time = 0.3
        self.action_duration = 0.2

        self.episode_step = 0
        self.begin_episode_step = np.floor(np.random.rand() * self.episode_length)
        self.last_action_timestamp = 0
        self.reverse_episode = False

        self.robot_vel = robot_vel
        self.object_vel = object_vel
        self.state_dict = {
            "robot": robot_pos,
            "object": object_pos,
            "corner": corner_pos,
        }
        self.action_timestamp = action_timestamp
        self.last_action_timestamp = self.action_timestamp.copy()

        self.action_publisher = rospy.Publisher("/action_topic", RobotCmd, queue_size=1)
        self.camera_img_subscriber = rospy.Subscriber(...)

        # define centers and radius for figure-8 trajectory
        rospy.sleep(0.2)        # wait for states to be published and set

        if self.robot_goals:
            back_circle_center_rel = np.array([0.7, 0.5])
            front_circle_center_rel = np.array([0.4, 0.5])
        else:
            back_circle_center_rel = np.array([0.65, 0.5])
            front_circle_center_rel = np.array([0.4, 0.5])

        corner_pos = self.state_dict["corner"].copy()
        self.back_circle_center = back_circle_center_rel * corner_pos[:2]
        self.front_circle_center = front_circle_center_rel * corner_pos[:2]
        self.radius = np.linalg.norm(self.back_circle_center - self.front_circle_center) / 2

        assert os.path.exists(YAW_OFFSET_PATH)
        self.yaw_offsets = np.load(YAW_OFFSET_PATH)

        # self.n_robots = n_robots
        # self.action_duration = action_duration
        # self.episode_length = episode_length

    def __getattr__(self, key):
        return self.params[key]

    def step(self, action, reset=False):
        action_msg = build_action_msg(action, self.action_duration, self.episode_step)
        self.action_publisher.publish(action_msg)

        # if self.last_action_timestamp == self.action_timestamp:
        rospy.wait_for_message("/action_timestamps", Time, timeout=5)

        rospy.sleep(self.action_duration + self.post_action_sleep_time)

        state = self.get_state()

        if not reset:
            self.episode_step += 1

        done = (self.episode_step == self.episode_length)

        return state, done

    def reset(self, replay_buffer):
        self.episode_step = 0
        self.begin_episode_step = np.floor(np.random.rand() * self.episode_length)

        init_goal = self.get_goal(step_override=0)
        state = self.get_state()

        while np.linalg.norm(init_goal - state) > self.init_tolerance:
            action = self.agent.get_action()
            next_state, _ = self.step(action, reset=True)

            if state and next_state:
                replay_buffer.add(state, action, next_state)

            state = next_state

        self.reverse_episode = not self.reverse_episode

        return state

    def render(self):
        # get current state
        # make plot based on relevant states and goals (track self.states or something)
        # get real camera image from a ros subscriber

        plot_img = ...
        real_img = ...

        ...

        return plot_img, real_img

    def get_state(self):
        robot_pos = self.state_dict["robot"].copy()
        object_pos = self.state_dict["object"].copy()
        corner_pos = self.state_dict["corner"].copy()

        # apply yaw offset
        robot_pos[2] = (robot_pos[2] + self.yaw_offsets[self.robot_id]) % (2 * np.pi)

        if self.use_object:
            object_pos[2] = (object_pos[2] + self.yaw_offsets[self.object_id]) % (2 * np.pi)
            state = np.concatenate((robot_pos, object_pos), axis=0)
        else:
            state = robot_pos

        out_of_bounds = lambda pos: np.any(pos[:2] > corner_pos[:2]) or np.any(pos[:2] < 0)

        valid = not out_of_bounds(robot_pos)

        if self.use_object:
            valid = valid and not out_of_bounds(object_pos)

        if not valid:
            print("\nOUT OF BOUNDS\n")
            import pdb;pdb.set_trace()
            state = None

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
