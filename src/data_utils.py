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

    def state_to_xysc(self, state, single_state=False):
        """
        Inputs:
            state = [x, y, theta, x, y, theta]
        Output:
            if single_state:
                state = [x, y, sin(theta), cos(theta)]
            else:
                state = [x, y, sin(theta), cos(theta), x, y, sin(theta), cos(theta)]
        """
        state = as_tensor(state)

        robot_xy, robot_theta = state[:, :2], state[:, 2]
        robot_sc = torch.stack((torch.sin(robot_theta), torch.cos(robot_theta)), dim=1)
        robot_xysc = torch.cat((robot_xy, robot_sc), dim=-1)

        if self.use_object and not single_state:
            object_xy, object_theta = state[:, 3:5], state[:, 5]
            object_sc = torch.stack((torch.sin(object_theta), torch.cos(object_theta)), dim=1)
            object_xysc = torch.cat((object_xy, object_sc), dim=-1)

            full_state = torch.cat((robot_xysc, object_xysc), dim=-1)
        else:
            full_state = robot_xysc

        return full_state

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
            full_state = torch.cat((robot_xy, robot_theta[:, None], object_xy, object_theta[:, None]), dim=1)
        else:
            full_state = torch.cat((robot_xy, robot_theta[:, None]), dim=1)

        return full_state

    def state_to_model_input(self, state, get_xysc=False):
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
        robot_sc = torch.stack((torch.sin(robot_theta), torch.cos(robot_theta)), dim=1)
        robot_xysc = torch.cat((robot_xy, robot_sc), dim=-1)

        if self.use_object:
            object_xy, object_theta = state[:, 3:5], state[:, 5]
            object_sc = torch.stack((torch.sin(object_theta), torch.cos(object_theta)), dim=1)
            object_xysc = torch.cat((object_xy, object_sc), dim=-1)

            obj_to_robot_xysc = robot_xysc - object_xysc
            full_state = torch.cat((robot_sc, obj_to_robot_xysc), dim=-1)

            if get_xysc:
                return full_state, robot_xysc, object_xysc
        else:
            full_state = robot_sc

            if get_xysc:
                return full_state, robot_xysc, robot_sc

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

        robot_state, robot_next_state = state[:, :3], next_state[:, :3]
        robot_xysc, robot_next_xysc = self.state_to_xysc(robot_state, single_state=True), self.state_to_xysc(robot_next_state, single_state=True)
        robot_state_delta_xysc = robot_next_xysc - robot_xysc

        if self.use_object:
            object_state, object_next_state = state[:, 3:], next_state[:, 3:]
            object_xysc, object_next_xysc = self.state_to_xysc(object_state, single_state=True), self.state_to_xysc(object_next_state, single_state=True)
            object_state_delta_xysc = object_next_xysc - object_xysc

            state_delta_xysc = torch.cat((robot_state_delta_xysc, object_state_delta_xysc), dim=-1)
        else:
            state_delta_xysc = robot_state_delta_xysc

        return state_delta_xysc

    def compute_next_state(self, state, state_delta):
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
