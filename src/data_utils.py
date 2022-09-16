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
    def __init__(self, use_object=True):
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
        robot_theta = torch.atan2(robot_sin, robot_cos)

        if self.use_object:
            object_xy, object_sin, object_cos = state[:, 4:6], state[:, 6], state[:, 7]
            object_theta = torch.atan2(object_sin, object_cos)

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
        robot_xysc = torch.cat((robot_xy, robot_sc), dim=1)

        if self.use_object:
            object_xy, object_theta = state[:, 3:5], state[:, 5]
            object_sc = torch.stack((torch.sin(object_theta), torch.cos(object_theta)), dim=1)
            object_xysc = torch.cat((object_xy, object_sc), dim=1)

            obj_to_robot_xysc = robot_xysc - object_xysc
            full_state = torch.cat((robot_sc, obj_to_robot_xysc), dim=1)

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

    def compute_next_state(self, state, state_delta, xysc=False):
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
        state_xysc = self.state_to_xysc(state)
        next_state_xysc = state_xysc + state_delta
        next_state = self.state_from_xysc(next_state_xysc)

        if xysc:
            return next_state, next_state_xysc
        else:
            return next_state
