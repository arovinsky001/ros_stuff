#!/usr/bin/python3

import numpy as np
import torch

# from ros_stuff.msg import RobotCmd


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

def signed_angle_difference(angle1, angle2):
    return (angle1 - angle2 + torch.pi) % (2 * torch.pi) - torch.pi

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

    def compute_relative_delta_xysc(self, state, next_state):
        state, next_state = as_tensor(state, next_state)

        robot_xy, next_robot_xy = state[:, :2], next_state[:, :2]
        robot_heading, next_robot_heading = state[:, 2], next_state[:, 2]

        # rotate xy deltas by -robot_heading to get xy deltas relative to current robot state
        sin, cos = torch.sin(-robot_heading), torch.cos(-robot_heading)
        rotation = torch.stack((torch.stack((cos, -sin)),
                                torch.stack((sin, cos)))).permute(2, 0, 1)
        absolute_robot_delta_xy = next_robot_xy - robot_xy
        relative_robot_delta_xy = (rotation @ absolute_robot_delta_xy[:, :, None]).squeeze()

        # compute sin and cos of next robot heading relative to current robot heading
        relative_next_robot_heading = signed_angle_difference(next_robot_heading, robot_heading)
        rel_next_sin, rel_next_cos = torch.sin(relative_next_robot_heading), torch.cos(relative_next_robot_heading)
        relative_robot_delta_sc = torch.stack((rel_next_sin, rel_next_cos), dim=1)

        relative_robot_delta_xysc = torch.cat((relative_robot_delta_xy, relative_robot_delta_sc), dim=1)

        if self.use_object:
            object_xy, next_object_xy = state[:, 3:5], next_state[:, 3:5]
            object_heading, next_object_heading = state[:, 5], next_state[:, 5]

            # compute object-to-robot xy deltas rotated by -robot_heading (all w.r.t. current robot state)
            abs_object_to_robot_xy = robot_xy - object_xy
            abs_next_object_to_robot_xy = robot_xy - next_object_xy
            absolute_object_to_robot_delta_xy = abs_next_object_to_robot_xy - abs_object_to_robot_xy
            relative_object_delta_xy = (rotation @ absolute_object_to_robot_delta_xy[:, :, None]).squeeze()

            # compute sin and cos difference from next to current object heading relative to robot
            relative_object_heading = signed_angle_difference(object_heading, robot_heading)
            relative_next_object_heading = signed_angle_difference(next_object_heading, robot_heading)

            rel_sin, rel_cos = torch.sin(relative_object_heading), torch.cos(relative_object_heading)
            rel_next_sin, rel_next_cos = torch.sin(relative_next_object_heading), torch.cos(relative_next_object_heading)
            rel_sin_diff, rel_cos_diff = rel_next_sin - rel_sin, rel_next_cos - rel_cos
            relative_object_delta_sc = torch.stack((rel_sin_diff, rel_cos_diff), dim=1)

            relative_object_delta_xysc = torch.cat((relative_object_delta_xy, relative_object_delta_sc), dim=1)
            relative_state_delta_xysc = torch.cat((relative_robot_delta_xysc, relative_object_delta_xysc), dim=1)
        else:
            relative_state_delta_xysc = relative_robot_delta_xysc

        return relative_state_delta_xysc

    def next_state_from_relative_delta(self, state, relative_state_delta):
        state, relative_state_delta = as_tensor(state, relative_state_delta)

        robot_xy, robot_heading = state[:, :2], state[:, 2]
        relative_robot_delta_xy = relative_state_delta[:, :2]
        relative_robot_delta_sc = relative_state_delta[:, 2:4]

        sin, cos = torch.sin(robot_heading), torch.cos(robot_heading)
        rotation = torch.stack((torch.stack((cos, -sin)),
                                torch.stack((sin, cos)))).permute(2, 0, 1)
        absolute_robot_delta_xy = (rotation @ relative_robot_delta_xy[:, :, None]).squeeze()
        absolute_next_robot_xy = robot_xy + absolute_robot_delta_xy

        rel_delta_sin, rel_delta_cos = relative_robot_delta_sc[:, 0], relative_robot_delta_sc[:, 1]
        absolute_robot_delta_heading = torch.atan2(rel_delta_sin, rel_delta_cos)
        absolute_next_robot_heading = (robot_heading + absolute_robot_delta_heading) % (2 * torch.pi)

        next_robot_state = torch.cat((absolute_next_robot_xy, absolute_next_robot_heading[:, None]), dim=1)

        if self.use_object:
            object_xy, object_heading = state[:, 3:5], state[:, 5]
            relative_object_delta_xy = relative_state_delta[:, 4:6]
            relative_object_delta_sc = relative_state_delta[:, 6:8]

            sin, cos = torch.sin(-robot_heading), torch.cos(-robot_heading)
            reverse_rotation = torch.stack((torch.stack((cos, -sin)),
                                            torch.stack((sin, cos)))).permute(2, 0, 1)
            absolute_object_to_robot_xy = robot_xy - object_xy
            relative_object_to_robot_xy = (reverse_rotation @ absolute_object_to_robot_xy[:, :, None]).squeeze()
            relative_next_object_to_robot_xy = relative_object_to_robot_xy + relative_object_delta_xy
            absolute_next_object_to_robot_xy = (rotation @ relative_next_object_to_robot_xy[:, :, None]).squeeze()
            absolute_next_object_xy = robot_xy - absolute_next_object_to_robot_xy

            relative_object_heading = signed_angle_difference(object_heading, robot_heading)
            rel_sin, rel_cos = torch.sin(relative_object_heading), torch.cos(relative_object_heading)
            relative_object_sc = torch.stack((rel_sin, rel_cos), dim=1)
            relative_next_object_sc = relative_object_sc + relative_object_delta_sc
            rel_next_sin, rel_next_cos = relative_next_object_sc[:, 0], relative_next_object_sc[:, 1]
            relative_next_object_heading = torch.atan2(rel_next_sin, rel_next_cos)
            absolute_next_object_heading = (relative_next_object_heading + robot_heading) % (2 * torch.pi)

            next_object_state = torch.cat((absolute_next_object_xy, absolute_next_object_heading[:, None]), dim=1)
            next_state = torch.cat((next_robot_state, next_object_state), dim=1)
        else:
            next_state = next_robot_state

        return next_state
