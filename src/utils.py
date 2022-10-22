#!/usr/bin/python3

import numpy as np
import torch

from ros_stuff.msg import RobotCmd


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

def apply_yaw_perturbations(state, action, next_state, state_delta):
    state, action, state_delta = state.clone(), action.clone(), state_delta.clone()

    offsets_per_sample = 40

    # generate random yaw offsets and add them to the robot heading
    heading_offset = torch.rand(offsets_per_sample) * 2 * torch.pi
    offset_state = state.repeat(1, offsets_per_sample).reshape(len(state), offsets_per_sample, -1)
    offset_state[:, :, 2] = (offset_state[:, :, 2] + heading_offset) % (2 * torch.pi)

    # create rotation matrix corresponding to heading offsets
    sin, cos = torch.sin(heading_offset), torch.cos(heading_offset)
    rotation = torch.stack((torch.stack((cos, -sin)),
                            torch.stack((sin, cos)))).transpose(0, 2)
    tmp = rotation[:, 0, 1].clone()
    rotation[:, 0, 1] = rotation[:, 1, 0].clone()
    rotation[:, 1, 0] = tmp

    # apply rotations to xy components of robot state_delta
    offset_state_delta = state_delta.repeat(1, offsets_per_sample).reshape(len(state_delta), offsets_per_sample, -1)
    offset_state_delta[:, :, :2] = (rotation @ offset_state_delta[:, :, :2, None]).squeeze()

    # compute offset sin and cos deltas for robot
    state_offset_heading = offset_state[:, :, 2].reshape(-1)
    next_state_repeat = next_state.repeat(1, offsets_per_sample).reshape(len(next_state), offsets_per_sample, -1)
    next_state_offset_heading = ((next_state_repeat[:, :, 2] + heading_offset) % (2 * torch.pi)).reshape(-1)
    offset_sin, offset_cos = torch.sin(state_offset_heading), torch.cos(state_offset_heading)
    next_offset_sin, next_offset_cos = torch.sin(next_state_offset_heading), torch.cos(next_state_offset_heading)
    sin_delta, cos_delta = next_offset_sin - offset_sin, next_offset_cos - offset_cos

    state = offset_state.reshape(-1, state.shape[-1])
    state_delta = offset_state_delta.reshape(-1, state_delta.shape[-1])
    action = action.repeat(1, offsets_per_sample).reshape(-1, action.shape[-1])

    state_delta[:, 2] = sin_delta
    state_delta[:, 3] = cos_delta

    return state, action, state_delta

def build_action_request(action, duration):
    action_req = RobotCmd()
    action_req.left_pwm = action[0]
    action_req.right_pwm = action[1]
    action_req.duration = duration
    return action_req


class DataUtils:
    def __init__(self, use_object=False):
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
