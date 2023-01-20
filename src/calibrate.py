#!/usr/bin/python3

import argparse
import numpy as np
import os

YAW_OFFSET_PATH = os.path.expanduser("~/kamigami_data/calibration/yaw_offsets.npy")


def main(args):
    yaw_offset_dir = "/".join(YAW_OFFSET_PATH.split("/")[:-1])
    if not os.path.exists(yaw_offset_dir):
        os.makedirs(yaw_offset_dir)

    for id in args.ids:
        yaw_offsets = np.zeros(10)

        input(f"Place tag {id} on the left calibration point, aligned with the calibration line and hit enter.")
        left_state = self.env.get_state()
        input(f"Place tag {id} on the right calibration point, aligned with the calibration line and hit enter.")
        right_state = self.env.get_state()

        robot_left_state, robot_right_state = left_state[:3], right_state[:3]
        true_robot_vector = (robot_left_state - robot_right_state)[:2]
        true_robot_angle = np.arctan2(true_robot_vector[1], true_robot_vector[0])
        measured_robot_angle = robot_left_state[2]
        yaw_offsets[self.robot_id] = true_robot_angle - measured_robot_angle

        if self.use_object:
            object_left_state, object_right_state = left_state[3:6], right_state[3:6]
            true_object_vector = (object_left_state - object_right_state)[:2]
            true_object_angle = np.arctan2(true_object_vector[1], true_object_vector[0])
            measured_object_angle = object_left_state[2]
            yaw_offsets[self.object_id] = true_object_angle - measured_object_angle

        np.save(self.yaw_offset_path, yaw_offsets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-ids", default=[0])

    args = parser.parse_args()
    main(args)