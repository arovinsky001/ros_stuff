#!/usr/bin/python

import argparse
import numpy as np
import os
import rospy

from environment import Environment
from utils import YAW_OFFSET_PATH, make_state_subscriber


def main(args):
    rospy.init_node("calibrate")

    yaw_offset_dir = "/".join(YAW_OFFSET_PATH.split("/")[:-1])
    if not os.path.exists(yaw_offset_dir):
        os.makedirs(yaw_offset_dir)

    params = {
        "episode_length": np.inf,
        "robot_goals": True,
        "robot_id": 0,
        "use_object": False,
    }

    robot_ids = args.ids[:-1]
    robot_pos, object_pos, corner_pos, robot_vel, object_vel, action_timestamp, tf_buffer, tf_listener = make_state_subscriber(robot_ids)
    env = Environment(robot_pos, object_pos, corner_pos, robot_vel, object_vel, action_timestamp, params, calibrate=True)
    yaw_offsets = np.zeros(10)

    for id in args.ids:
        input(f"Place tag {id} on the left calibration point, aligned with the calibration line and hit enter.")
        left_state = env.get_state()
        input(f"Place tag {id} on the right calibration point, aligned with the calibration line and hit enter.")
        right_state = env.get_state()

        robot_left_state, robot_right_state = left_state[:3], right_state[:3]
        true_robot_vector = (robot_left_state - robot_right_state)[:2]
        true_robot_angle = np.arctan2(true_robot_vector[1], true_robot_vector[0])
        measured_robot_angle = robot_left_state[2]
        yaw_offsets[id] = true_robot_angle - measured_robot_angle

        # if self.use_object:
        #     object_left_state, object_right_state = left_state[3:6], right_state[3:6]
        #     true_object_vector = (object_left_state - object_right_state)[:2]
        #     true_object_angle = np.arctan2(true_object_vector[1], true_object_vector[0])
        #     measured_object_angle = object_left_state[2]
        #     yaw_offsets[self.object_id] = true_object_angle - measured_object_angle

    np.save(YAW_OFFSET_PATH, yaw_offsets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-ids", nargs='+', type=int)

    args = parser.parse_args()
    main(args)
