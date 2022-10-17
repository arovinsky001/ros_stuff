import numpy as np
import torch
from pdb import set_trace

# GENERAL PYTORCH UTILS

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


# MODEL INPUT/OUTPUT UTILS

class DataUtils:
    def __init__(self, use_object=False):
        self.use_object = use_object

    def state_to_xysc(self, state):
        """
        Inputs:
            if use_object:
                state = [x, y, yaw]
            else:
                state = [x, y, yaw, x, y, yaw]
        Output:
            if use_object:
                state = [x, y, sin(yaw), cos(yaw)]
            else:
                state = [x, y, sin(yaw), cos(yaw), x, y, sin(yaw), cos(yaw)]
        """
        state = as_tensor(state)

        robot_xy, robot_yaw = state[:, :2], state[:, 2]
        robot_sc = torch.stack((torch.sin(robot_yaw), torch.cos(robot_yaw)), dim=1)
        robot_xysc = state_xysc = torch.cat((robot_xy, robot_sc), dim=1)

        if self.use_object:
            assert state.shape[1] == 6
            object_xy, object_yaw = state[:, 3:5], state[:, 5]
            object_sc = torch.stack((torch.sin(object_yaw), torch.cos(object_yaw)), dim=1)
            object_xysc = torch.cat((object_xy, object_sc), dim=1)

            state_xysc = torch.cat((robot_xysc, object_xysc), dim=1)
        else:
            assert state.shape[1] == 3

        return state_xysc

    def state_from_xysc(self, state):
        """
        Inputs:
            state = [x, y, sin(yaw), cos(yaw)]
        Output:
            state = [x, y, yaw]
        """
        state = as_tensor(state)

        robot_xy, robot_sin, robot_cos = state[:, :2], state[:, 2], state[:, 3]
        robot_yaw = torch.atan2(robot_sin, robot_cos) % (2 * torch.pi)

        if self.use_object:
            assert state.shape[1] == 8
            object_xy, object_sin, object_cos = state[:, 4:6], state[:, 6], state[:, 7]
            object_yaw = torch.atan2(object_sin, object_cos) % (2 * torch.pi)

            state_xyt = torch.cat((robot_xy, robot_yaw[:, None], object_xy, object_yaw[:, None]), dim=1)
        else:
            assert state.shape[1] == 4
            state_xyt = torch.cat((robot_xy, robot_yaw[:, None]), dim=1)

        return state_xyt

    def state_to_model_input(self, state):
        """
        Inputs:
            if use_object:
                state = [robot_x, robot_y, robot_yaw, object_x, object_y, object_yaw]
            else:
                state = [robot_x, robot_y, robot_yaw]
        Output:
            if use_object:
                state = [sin(robot_yaw), cos(robot_yaw), x_to_robot, y_to_robot, yaw_to_robot]
            else:
                state = [sin(robot_yaw), cos(robot_yaw)]
        """
        state = as_tensor(state)

        robot_state = state[:, :3]
        robot_sc = torch.stack((torch.sin(robot_state[:, 2]), torch.cos(robot_state[:, 2])), dim=1)

        if self.use_object:
            object_state = state[:, 3:6]
            object_to_robot = robot_state - object_state
            object_to_robot[:, 2] = (object_to_robot[:, 2] + torch.pi) % (2 * torch.pi) - torch.pi

            full_state = torch.cat((robot_sc, object_to_robot), dim=1)
        else:
            full_state = robot_sc

        return full_state

    def compute_state_delta(self, state, next_state):
        """
        Inputs:
            state = [x, y, yaw]
            next_state = [x, y, yaw]
        Output:
            state_delta = [x, y, sin(yaw), cos(yaw)]
        """
        state, next_state = as_tensor(state, next_state)
        state_delta = next_state - state
        state_delta[:, 2] = (state_delta[:, 2] + torch.pi) % (2 * torch.pi) - torch.pi

        if self.use_object:
            state_delta[:, 5] = (state_delta[:, 5] + torch.pi) % (2 * torch.pi) - torch.pi

        return state_delta

    def compute_next_state(self, state, state_delta):
        """
        Inputs:
            if use_object:
                state = [x, y, yaw, x, y, yaw]
                state_delta = [x, y, sin(yaw), cos(yaw), x, y, sin(yaw), cos(yaw)]
            else:
                state = [x, y, yaw]
                state_delta = [x, y, sin(yaw), cos(yaw)]
        Output:
            if use_object:
                next_state = [x, y, yaw, x, y, yaw]
            else:
                next_state = [x, y, yaw]
        """
        state, state_delta = as_tensor(state, state_delta)
        state_xysc = self.state_to_xysc(state)
        next_state_xysc = state_xysc + state_delta
        next_state = self.state_from_xysc(next_state_xysc)

        return next_state
