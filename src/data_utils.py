import numpy as np
import torch

# GENERAL PYTORCH UTILS

device = torch.device("cpu")
def to_device(*args, device=device):
    ret = []
    for arg in args:
        ret.append(arg.to(device))
    return ret if len(ret) > 1 else ret[0]

def dcn(*args):
    ret = []
    for arg in args:
        ret.append(arg.detach().cpu().numpy())
    return ret if len(ret) > 1 else ret[0]

def as_tensor(*args):
    ret = []
    for arg in args:
        if type(arg) == np.ndarray:
            ret.append(torch.as_tensor(arg, dtype=torch.float))
        else:
            ret.append(arg)
    return ret if len(ret) > 1 else ret[0]


# MODEL INPUT/OUTPUT UTILS

def convert_state(state, get_xysc=False, use_object=True):
    state = as_tensor(state)

    if use_object:
        robot_xy, robot_theta = state[:, :2], state[:, 2]
        robot_sc = torch.stack((torch.sin(robot_theta), torch.cos(robot_theta)), dim=1)
        robot_xysc = torch.cat((robot_xy, robot_sc), dim=-1)

        object_xy, object_theta = state[:, 3:5], state[:, 5]
        object_sc = torch.stack((torch.sin(object_theta), torch.cos(object_theta)), dim=1)
        object_xysc = torch.cat((object_xy, object_sc), dim=-1)

        obj_to_robot_xysc = robot_xysc - object_xysc
        full_state = torch.cat((robot_sc, obj_to_robot_xysc), dim=-1)

        if get_xysc:
            return full_state, robot_xysc, object_xysc
    else:
        robot_xy, robot_theta = state[:, :2], state[:, 2]
        robot_sc = torch.stack((torch.sin(robot_theta), torch.cos(robot_theta)), dim=1)
        robot_xysc = torch.cat((robot_xy, robot_sc), dim=-1)
        full_state = robot_sc

        if get_xysc:
            return full_state, robot_xysc, robot_sc

    return full_state

def convert_state_delta(state, next_state, use_object=True):
    next_state = as_tensor(next_state)

    if use_object:
        full_state, robot_xysc, object_xysc = convert_state(state, get_xysc=True, use_object=True)

        robot_next_xy, robot_next_theta = next_state[:, :2], next_state[:, 2]
        robot_next_sc = torch.stack([torch.sin(robot_next_theta), torch.cos(robot_next_theta)], dim=1)
        robot_next_xysc = torch.cat([robot_next_xy, robot_next_sc], dim=-1)

        object_next_xy, object_next_theta = next_state[:, 3:5], next_state[:, 5]
        object_next_sc = torch.stack([torch.sin(object_next_theta), torch.cos(object_next_theta)], dim=1)
        object_next_xysc = torch.cat([object_next_xy, object_next_sc], dim=-1)

        robot_state_delta = robot_next_xysc - robot_xysc
        object_state_delta = object_next_xysc - object_xysc
        state_delta = torch.cat((robot_state_delta, object_state_delta), dim=-1)
    else:
        full_state, robot_xysc, robot_sc = convert_state(state, get_xysc=True, use_object=False)

        robot_next_xy, robot_next_theta = next_state[:, :2], next_state[:, 2]
        robot_next_sc = torch.stack([torch.sin(robot_next_theta), torch.cos(robot_next_theta)], dim=1)
        robot_next_xysc = torch.cat([robot_next_xy, robot_next_sc], dim=-1)

        state_delta = robot_next_xysc - robot_xysc
        full_state = robot_sc

    return full_state, state_delta

def compute_next_state(state, state_delta):
    """
    Inputs:
        state = [x, y, theta]
        state_delta = [x, y, sin(theta), cos(theta)]
    Output:
        next_state = [x, y, theta]
    """
    xy, theta = state[:, :-1], state[:, -1]
    sc = torch.stack((torch.sin(theta), torch.cos(theta)), dim=1)
    xysc = torch.cat((xy, sc), dim=1)
    next_xysc = xysc + state_delta
    next_xy, next_sin, next_cos = next_xysc[:, :2], next_xysc[:, 2], next_xysc[:, 3]
    next_theta = torch.atan2(next_sin, next_cos)
    next_state = torch.cat((next_xy, next_theta[:, None]), dim=-1)
    return next_state
