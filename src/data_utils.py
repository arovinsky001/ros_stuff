#!/usr/bin/python3

import numpy as np
import torch


dimensions = {
    "action_dim": 2,                            # left_pwm, right_pwm
    "state_dim": 3,                             # x, y, yaw
    "robot_input_dim": 2,                       # sin(yaw), cos(yaw)
    "object_input_dim": 3,                      # x_to_robot, y_to_robot, yaw_to_robot
    "robot_output_dim": 3,                      # x_delta, y_delta, yaw_delta
    "object_output_dim": 3,                     # x_delta, y_delta, yaw_delta
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
    def __init__(self, use_object=False):
        self.use_object = use_object

    def state_to_model_input(self, state):
        state = as_tensor(state)
        robot_state = state[:, :dimensions["state_dim"]]

        yaw_idx = dimensions["state_dim"] - 1
        robot_sc = torch.stack((torch.sin(robot_state[:, yaw_idx]), torch.cos(robot_state[:, yaw_idx])), dim=1)

        if self.use_object:
            object_state = state.chunk(2, dim=1)[1]
            object_to_robot = robot_state - object_state
            object_to_robot[:, yaw_idx] = signed_angle_difference(object_to_robot[:, yaw_idx])

            full_state = torch.cat((robot_sc, object_to_robot), dim=1)
        else:
            full_state = robot_sc

        return full_state

    def compute_state_delta(self, state, next_state):
        state, next_state = as_tensor(state, next_state)
        state_delta = next_state - state

        yaw_idx = dimensions["state_dim"] - 1
        state_delta[:, yaw_idx] = signed_angle_difference(state_delta[:, yaw_idx])

        if self.use_object:
            yaw_idx += dimensions["state_dim"]
            state_delta[:, yaw_idx] = signed_angle_difference(state_delta[:, yaw_idx])

        return state_delta

    def compute_next_state(self, state, state_delta):
        state, state_delta = as_tensor(state, state_delta)
        next_state = state + state_delta

        yaw_idx = dimensions["state_dim"] - 1
        next_state[:, yaw_idx] = next_state[:, yaw_idx] % (2 * torch.pi)

        if self.use_object:
            yaw_idx += dimensions["state_dim"]
            next_state[:, yaw_idx] = next_state[:, yaw_idx] % (2 * torch.pi)

        return next_state
