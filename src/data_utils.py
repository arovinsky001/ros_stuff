#!/usr/bin/python3

import numpy as np
import torch


dimensions = {
    "action_dim": 2,                            # left_pwm, right_pwm
    "state_dim": 3,                             # x, y, yaw
    "robot_input_dim": 2,                       # sin(yaw), cos(yaw)
    "object_input_dim": 4,                      # x_to_robot, y_to_robot, sin(object_yaw), cos(object_yaw)
    "robot_output_dim": 4,                      # x_delta, y_delta, sin(robot_yaw), cos(robot_yaw)
    "object_output_dim": 4,                     # x_delta, y_delta, sin(object_yaw), cos(object_yaw)
}

### GENERAL PYTORCH UTILS ###

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

def signed_angle_difference(diff):
    return (diff + torch.pi) % (2 * torch.pi) - torch.pi


class DataUtils:
    def __init__(self, use_object=False, robot_theta=False):
        self.use_object = use_object

    def state_to_xysc(self, state):
        """
        Inputs:
            if use_object:
                state = [x, y, theta]
            else:
                state = [x, y, theta, x, y, theta]
        Output:
            if use_object:
                state = [x, y, sin(theta), cos(theta)]
            else:
                state = [x, y, sin(theta), cos(theta), x, y, sin(theta), cos(theta)]
        """
        state = as_tensor(state)

        robot_xy, robot_theta = state[:, :2], state[:, 2]
        robot_sc = torch.stack((torch.sin(robot_theta), torch.cos(robot_theta)), dim=1)
        robot_xysc = state_xysc = torch.cat((robot_xy, robot_sc), dim=1)

        if self.use_object:
            object_xy, object_theta = state[:, 3:5], state[:, 5]
            object_sc = torch.stack((torch.sin(object_theta), torch.cos(object_theta)), dim=1)
            object_xysc = torch.cat((object_xy, object_sc), dim=1)

            state_xysc = torch.cat((robot_xysc, object_xysc), dim=1)

        return state_xysc

    def state_from_xysc(self, state):
        """
        Inputs:
            state = [x, y, sin(theta), cos(theta)]
        Output:
            state = [x, y, theta]
        """
        state = as_tensor(state)

        robot_xy, robot_sin, robot_cos = state[:, :2], state[:, 2], state[:, 3]
        robot_theta = torch.atan2(robot_sin, robot_cos) % (2 * torch.pi)

        if self.use_object:
            object_xy, object_sin, object_cos = state[:, 4:6], state[:, 6], state[:, 7]
            object_theta = torch.atan2(object_sin, object_cos) % (2 * torch.pi)

            state_xyt = torch.cat((robot_xy, robot_theta[:, None], object_xy, object_theta[:, None]), dim=1)
        else:
            state_xyt = torch.cat((robot_xy, robot_theta[:, None]), dim=1)

        return state_xyt

    def state_to_model_input(self, state):
        """
        Inputs:
            if use_object:
                state = [x, y, theta, x, y, theta]
            else:
                state = [x, y, theta]
        Output:
            if use_object:
                state = [sin(theta), cos(theta), x, y, sin(theta), cos(theta)]
            else:
                state = [sin(theta), cos(theta)]
        """
        state = as_tensor(state)

        robot_xy, robot_theta = state[:, :2], state[:, 2]
        robot_sc = full_state = torch.stack((torch.sin(robot_theta), torch.cos(robot_theta)), dim=1)

        if self.use_object:
            object_xy, object_theta = state[:, 3:5], state[:, 5]
            obj_to_robot_xy = robot_xy - object_xy
            object_sc = torch.stack((torch.sin(object_theta), torch.cos(object_theta)), dim=1)
            full_state = torch.cat((robot_sc, obj_to_robot_xy, object_sc), dim=1)

        return full_state

    def state_delta_xysc(self, state, next_state):
        """
        Inputs:
            state = [x, y, theta]
            next_state = [x, y, theta]
        Output:
            state_delta = [x, y, sin(theta), cos(theta)]
        """
        state, next_state = as_tensor(state, next_state)
        state_xysc, next_state_xysc = self.state_to_xysc(state), self.state_to_xysc(next_state)

        robot_xysc, robot_next_xysc = state_xysc[:, :4], next_state_xysc[:, :4]
        robot_state_delta_xysc = state_delta_xysc = robot_next_xysc - robot_xysc

        if self.use_object:
            object_xysc, object_next_xysc = state_xysc[:, 4:], next_state_xysc[:, 4:]
            object_state_delta_xysc = object_next_xysc - object_xysc

            state_delta_xysc = torch.cat((robot_state_delta_xysc, object_state_delta_xysc), dim=1)

        return state_delta_xysc

    def compute_next_state(self, state, state_delta):
        """
        Inputs:
            if use_object:
                state = [x, y, theta, x, y, theta]
                state_delta = [x, y, sin(theta), cos(theta), x, y, sin(theta), cos(theta)]
            else:
                state = [x, y, theta]
                state_delta = [x, y, sin(theta), cos(theta)]
        Output:
            if use_object:
                next_state = [x, y, theta, x, y, theta]
            else:
                next_state = [x, y, theta]
        """
        state, state_delta = as_tensor(state, state_delta)
        state_xysc = self.state_to_xysc(state)

        next_state_xysc = state_xysc + state_delta
        next_state = self.state_from_xysc(next_state_xysc)

        return next_state
