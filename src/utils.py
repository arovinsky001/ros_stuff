#!/usr/bin/python3

import os
import numpy as np
import torch

import rospy
import tf2_ros
from std_msgs.msg import Int8
from ros_stuff.msg import RobotCmd, MultiRobotCmd, ProcessedStates


YAW_OFFSET_PATH = os.path.expanduser("~/kamigami_data/calibration/yaw_offsets.npy")


### TORCH UTILS ###

def to_device(*args, device=torch.device("cpu")):
    ret = []
    for arg in args:
        ret.append(arg.to(device))
    return ret if len(ret) > 1 else ret[0]

def dcn(*args):
    ret = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            ret.append(arg)
        else:
            ret.append(arg.detach().cpu().numpy())
    return ret if len(ret) > 1 else ret[0]

def as_tensor(*args):
    ret = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            ret.append(torch.as_tensor(arg, dtype=torch.float))
        else:
            ret.append(arg)
    return ret if len(ret) > 1 else ret[0]

def sin_cos(angle):
    return torch.sin(angle), torch.cos(angle)

def signed_angle_difference(angle1, angle2):
    return (angle1 - angle2 + 3 * torch.pi) % (2 * torch.pi) - torch.pi

def rotate_vectors(vectors, angle):
    sin, cos = sin_cos(angle)
    rotation = torch.stack((torch.stack((cos, -sin)),
                            torch.stack((sin, cos)))).permute(2, 0, 1)
    rotated_vector = (rotation @ vectors[:, :, None]).squeeze(dim=-1)
    return rotated_vector


### ROS UTILS ###

def build_action_msg(action, duration, robot_ids):
    multi_action_msg = MultiRobotCmd()
    multi_action_msg.robot_commands = []

    action = np.clip(action, -1, 1)
    action_list = np.split(action, len(action) / 2.)

    for action, id in zip(action_list, robot_ids):
        action_msg = RobotCmd()
        action_msg.robot_id = id
        action_msg.left_pwm = action[0]
        action_msg.right_pwm = action[1]
        action_msg.duration = duration

        multi_action_msg.robot_commands.append(action_msg)

    return multi_action_msg

def make_state_subscriber(robot_ids):
    pos_dim = 3
    vel_dim = 3

    robot_pos_dict = {id: np.empty(pos_dim) for id in robot_ids}
    robot_vel_dict = {id: np.empty(vel_dim) for id in robot_ids}

    object_pos = np.empty(pos_dim)
    object_vel = np.empty(vel_dim)

    corner_pos = np.empty(pos_dim)

    def state_callback(msg):
        robot_states, os, cs = msg.robot_states, msg.object_state, msg.corner_state

        for rs in robot_states:
            robot_pos_dict[rs.robot_id][:] = np.array([rs.x, rs.y, rs.yaw])
            robot_vel_dict[rs.robot_id][:] = np.array([rs.x_vel, rs.y_vel, rs.yaw_vel])

        object_pos[:] = np.array([os.x, os.y, os.yaw])
        object_vel[:] = np.array([os.x_vel, os.y_vel, os.yaw_vel])

        corner_pos[:] = np.array([cs.x, cs.y, cs.yaw])

    print("waiting for /processed_state topic from state publisher")
    rospy.Subscriber("/processed_state", ProcessedStates, state_callback, queue_size=1)
    print("subscribed to /processed_state")

    # get action receipt from kamigami
    action_receipt_dict = {id: False for id in robot_ids}

    def action_receipt_callback(msg):
        action_receipt_dict[msg.data] = True

    for id in robot_ids:
        print(f"waiting for /action_receipt topic from kamigami {id}")
        rospy.Subscriber(f"/action_receipt_{id}", Int8, action_receipt_callback, queue_size=1)
        print(f"subscribed to /action_receipt_{id}")

    print("setting up tf buffer/listener")
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    print("finished setting up tf buffer/listener")

    return robot_pos_dict, robot_vel_dict, object_pos, object_vel, corner_pos, action_receipt_dict, tf_buffer, tf_listener


class DataUtils:
    def __init__(self, params):
        self.params = params

    def __getattr__(self, key):
        return self.params[key]

    def state_to_model_input(self, state):
        if self.n_robots == 1 and not self.use_object:
            return None

        state = as_tensor(state)

        n_states = self.n_robots + self.use_object
        relative_states = torch.empty((state.size(0), 4*(n_states-1)))
        base_xy, base_heading = state[:, :2], state[:, 2]

        for i in range(1, n_states):
            cur_state = state[:, 3*i:3*(i+1)]
            xy, heading = cur_state[:, :2], cur_state[:, 2]

            absolute_state_to_base_xy = base_xy - xy
            relative_state_to_base_xy = rotate_vectors(absolute_state_to_base_xy, -base_heading)

            relative_state_heading = signed_angle_difference(heading, base_heading)
            relative_state_sc = torch.stack(sin_cos(relative_state_heading), dim=1)

            relative_state = torch.cat((relative_state_to_base_xy, relative_state_sc), dim=1)
            relative_states[:, 4*(i-1):4*i] = relative_state

        return relative_states

    def compute_relative_delta_xysc(self, state, next_state):
        state, next_state = as_tensor(state, next_state)

        n_states = self.n_robots + self.use_object
        relative_delta_xysc = torch.empty((state.size(0), 4*n_states))
        base_heading = state[:, 2]

        for i in range(n_states):
            cur_state, cur_next_state = state[:, 3*i:3*(i+1)], next_state[:, 3*i:3*(i+1)]

            xy, next_xy = cur_state[:, :2], cur_next_state[:, :2]
            heading, next_heading = cur_state[:, 2], cur_next_state[:, 2]

            absolute_delta_xy = next_xy - xy
            relative_delta_xy = rotate_vectors(absolute_delta_xy, -base_heading)

            relative_heading = signed_angle_difference(heading, base_heading)
            relative_next_heading = signed_angle_difference(next_heading, base_heading)
            rel_sin, rel_cos = sin_cos(relative_heading)
            rel_next_sin, rel_next_cos = sin_cos(relative_next_heading)
            relative_delta_sc = torch.stack((rel_next_sin - rel_sin, rel_next_cos - rel_cos), dim=1)

            relative_delta_xysc[:, 4*i:4*(i+1)] = torch.cat((relative_delta_xy, relative_delta_sc), dim=1)

        return relative_delta_xysc

    def next_state_from_relative_delta(self, state, relative_delta):
        state, relative_delta = as_tensor(state, relative_delta)

        n_states = self.n_robots + self.use_object
        next_state = torch.empty_like(state)
        base_heading = state[:, 2]

        for i in range(n_states):
            cur_state, cur_relative_delta = state[:, 3*i:3*(i+1)], relative_delta[:, 4*i:4*(i+1)]

            absolute_xy, absolute_heading = cur_state[:, :2], cur_state[:, 2]
            relative_delta_xy, relative_delta_sc = cur_relative_delta[:, :2], cur_relative_delta[:, 2:]

            absolute_delta_xy = rotate_vectors(relative_delta_xy, base_heading)
            absolute_next_xy = absolute_xy + absolute_delta_xy

            relative_heading = signed_angle_difference(absolute_heading, base_heading)
            relative_sc = torch.stack(sin_cos(relative_heading), dim=1)
            relative_next_sc = relative_sc + relative_delta_sc
            rel_next_sin, rel_next_cos = relative_next_sc.T
            relative_next_heading = torch.atan2(rel_next_sin, rel_next_cos)
            absolute_next_heading = signed_angle_difference(relative_next_heading, -base_heading)

            next_state[:, 3*i:3*(i+1)] = torch.cat((absolute_next_xy, absolute_next_heading[:, None]), dim=1)

        return next_state

    def cost_dict(self, state, action, goals, robot_goals=False, signed=False):
        state_dim = 3
        if self.use_object:
            robot_state = state[..., :state_dim]
            object_state = state[..., -state_dim:]

            effective_state = robot_state if robot_goals else object_state
        else:
            effective_state = state[..., :state_dim]

        # distance to goal position
        state_to_goal_xy = (goals - effective_state)[..., :-1]
        dist_cost = np.linalg.norm(state_to_goal_xy, axis=-1)
        if signed:
            dist_cost *= forward

        # difference between current and toward-goal heading
        current_angle = effective_state[..., 2]
        target_angle = np.arctan2(state_to_goal_xy[..., 1], state_to_goal_xy[..., 0])
        heading_cost = signed_angle_difference(target_angle, current_angle)

        left = (heading_cost > 0) * 2 - 1
        forward = (np.abs(heading_cost) < np.pi / 2) * 2 - 1

        heading_cost[forward == -1] = (heading_cost[forward == -1] + np.pi) % (2 * np.pi)
        heading_cost = np.stack((heading_cost, 2 * np.pi - heading_cost)).min(axis=0)

        if signed:
            heading_cost *= left * forward
        else:
            heading_cost = np.abs(heading_cost)

        # difference between current and specified heading
        target_angle = goals[..., 2]
        target_heading_cost = signed_angle_difference(target_angle, current_angle)

        left = (target_heading_cost > 0) * 2 - 1
        forward = (np.abs(target_heading_cost) < np.pi / 2) * 2 - 1

        target_heading_cost[forward == -1] = (target_heading_cost[forward == -1] + np.pi) % (2 * np.pi)
        target_heading_cost = np.stack((target_heading_cost, 2 * np.pi - target_heading_cost)).min(axis=0)

        if signed:
            target_heading_cost *= left * forward
        else:
            target_heading_cost = np.abs(target_heading_cost)

        # object-robot separation distance
        if self.use_object:
            object_to_robot_xy = (robot_state - object_state)[..., :-1]
            sep_cost = np.linalg.norm(object_to_robot_xy, axis=-1)
        else:
            sep_cost = np.array([0.])

        # object-robot heading difference
        if self.use_object:
            robot_theta, object_theta = robot_state[..., -1], object_state[..., -1]
            heading_diff = (robot_theta - object_theta) % (2 * np.pi)
            heading_diff_cost = np.stack((heading_diff, 2 * np.pi - heading_diff), axis=1).min(axis=1)
        else:
            heading_diff_cost = np.array([0.])

        # action magnitude
        norm_cost = -np.linalg.norm(action, axis=-1)

        cost_dict = {
            "distance": dist_cost,
            "heading": heading_cost,
            "target_heading": target_heading_cost,
            "separation": sep_cost,
            "heading_difference": heading_diff_cost,
            "action_norm": norm_cost,
        }

        return cost_dict
