#!/usr/bin/python3
import matplotlib.pyplot as plt
from matplotlib import animation as anim, cm
import rospy
import numpy as np
import collections
from pdb import set_trace

from std_msgs.msg import Header
from ros_stuff.msg import ImuData

class Laptop():
    def __init__(self, robot_ids, plot=False):
        rospy.init_node("laptop")
        rospy.Subscriber("/imu_data", ImuData, self.callback, queue_size=1)
        print("IMU subscriber initialized")
        self.plot = plot
        self.robots = robot_ids
        self.num_robots = len(self.robots)
        self.ac_num = [-1 for i in range(self.num_robots)]
        # self.step_finished = [True for i in range(self.num_robots)]
        self.first_timestamp = [None for i in range(self.num_robots)]
        self.latest_timestamp = [None for i in range(self.num_robots)]
        self.began = [False for i in range(self.num_robots)]
        # map for robot_id to data index
        self.robot_idx_map = {self.robots[i]: i for i in range(self.num_robots)}

        # databases and buffers for each robot
        self.data = [{} for i in range(self.num_robots)]
        self.proc_data = [{} for i in range(self.num_robots)]
        self.ax_buffer = [[] for i in range(self.num_robots)]
        self.ay_buffer = [[] for i in range(self.num_robots)]
        self.mx_buffer = [[] for i in range(self.num_robots)]
        self.my_buffer = [[] for i in range(self.num_robots)]

        # self.a_thresh = 0.5

        if self.plot:
            self.fig = plt.figure(figsize=(12,8))
            self.figx = plt.subplot(221)
            self.fig_proc_x = plt.subplot(222)
            self.figy = plt.subplot(223)
            self.fig_proc_y = plt.subplot(224)
            plt.show()

        rospy.spin()

    def callback(self, msg):
        """
        For each data point published from the robot:
            1) determine that it is data corresponding to the appropriate step
                and corresponding to physical motion(above threshold value)
            2) append to buffers
            3) if this data point corresponds to near static behavior(below threshold),
                collect last datapoint timestamp, prepare class variables for next step,
                and process data
            TODO: moving avg percent change threshold
        """
        robot_id = int(msg.robot_id)
        try:
            robot_idx = self.robot_idx_map[robot_id]
        except:
            print("robot_id: ", robot_id)
            print("map: ", self.robot_idx_map)
        ac_num = msg.action_num
        if self.ac_num[robot_idx] == -1 and ac_num == 0 and not self.began[robot_idx]:
            # first msg of first step
            self.began[robot_idx] = True
            return

        if ac_num == (self.ac_num[robot_idx] + 1) and self.began[robot_idx]:
            if ac_num > 0:
                # process previous step
                last_timestamp = self.latest_timestamp[robot_idx]
                self.update_step(robot_idx)
                delta_t = last_timestamp - self.first_timestamp[robot_idx]
                self.process_data(delta_t, self.ac_num[robot_idx], robot_idx, plot=self.plot)

            # beginning of a new step
            print("new step!")
            self.ac_num[robot_idx] = ac_num
            self.first_timestamp[robot_idx] = msg.header.stamp.to_sec()

        self.ax_buffer[robot_idx].append(msg.a_x)
        self.ay_buffer[robot_idx].append(msg.a_y)
        self.mx_buffer[robot_idx].append(msg.m_x)
        self.my_buffer[robot_idx].append(msg.m_y)
        self.latest_timestamp[robot_idx] = msg.header.stamp.to_sec()

        # arctan2 the magnetometer data to find angle
        theta = np.arctan2(msg.m_y, msg.m_x)

    def process_data(self, delta_t, ac_num, robot_idx, plot=False):
        """
        Filter and Transform data, plot if necessary
        """
        data = self.data[robot_idx][ac_num]
        ax = data[0]
        ay = data[1]
        mx = data[2]
        my = data[3]
        # if delta_t != 0:
        #     sample_rate = len(ax) / delta_t
        # else:
        #     set_trace()
        # TODO: filter and transform

        if plot:
            self.figx.cla()
            self.figy.cla()
            print("ax:", ax)
            x = np.arange(len(ax))
            ##### x-axis data plot #####
            self.figx.plot(x, ax)
            self.figx.plot(x, mx)
            self.figx.set_ylim(-25,25)

            ##### y-axis data plot #####
            self.figy.plot(x, ay)
            self.figy.plot(x, my)
            self.figy.set_ylim(-25,25)
            # TODO: plot processed data
            plt.pause(0.001)
        return

    def update_step(self, robot_idx):
        """
        Store action-specific data to the 'data' dict and clear buffers
        """
        # store buffers
        ax = np.array(self.ax_buffer[robot_idx])
        ay = np.array(self.ay_buffer[robot_idx])
        mx = np.array(self.mx_buffer[robot_idx])
        my = np.array(self.my_buffer[robot_idx])
        ac_num = self.ac_num[robot_idx]
        self.data[robot_idx][ac_num] = np.vstack((ax, ay, mx, my))
        # reset buffers
        self.ax_buffer[robot_idx] = []
        self.ay_buffer[robot_idx] = []
        self.mx_buffer[robot_idx] = []
        self.my_buffer[robot_idx] = []

    def get_step_data(self, robot_idx, ac_num, processed=False):
        """
        Retrieve data corresponding to a specific step
        """
        if processed:
            # TODO: return processed step
            return
        if self.data[robot_idx].get(ac_num) is None:
            print(f"Action {ac_num} exceeds data buffer.")
            return None

        return self.data[robot_idx][ac_num]

    # def old_plotting_func(self, msg):
    #     # update deques
    #     self.aX.popleft()
    #     self.aX.append(msg.a_x)
    #     self.aY.popleft()
    #     self.aY.append(msg.a_y)
    #     self.aZ.popleft()
    #     self.aZ.append(msg.a_z)

    #     self.gX.popleft()
    #     self.gX.append(msg.g_x)
    #     self.gY.popleft()
    #     self.gY.append(msg.g_y)
    #     self.gZ.popleft()
    #     self.gZ.append(msg.g_z)

    #     # clear axes
    #     self.figx.cla()
    #     self.figy.cla()
    #     self.figz.cla()

    #     # === plot X ===
    #     self.figx.plot(self.aX)
    #     self.figx.scatter(len(self.aX)-1, self.aX[-1])
    #     self.figx.text(len(self.aX)-1, self.aX[-1]+2, "{}%".format(self.aX[-1]))

    #     self.figx.plot(self.gX)
    #     self.figx.scatter(len(self.gX)-1, self.gX[-1])
    #     self.figx.text(len(self.gX)-1, self.gX[-1]+2, "{}%".format(self.gX[-1]))

    #     self.figx.set_ylim(-25,25)

    #     # === plot Y ===
    #     self.figy.plot(self.aY)
    #     self.figy.scatter(len(self.aY)-1, self.aY[-1])
    #     self.figy.text(len(self.aY)-1, self.aY[-1]+2, "{}%".format(self.aY[-1]))

    #     self.figy.plot(self.gY)
    #     self.figy.scatter(len(self.gY)-1, self.gY[-1])
    #     self.figy.text(len(self.gY)-1, self.gY[-1]+2, "{}%".format(self.gY[-1]))

    #     self.figy.set_ylim(-25,25)

    #     # === plot Z ===
    #     self.figz.plot(self.aZ)
    #     self.figz.scatter(len(self.aZ)-1, self.aZ[-1])
    #     self.figz.text(len(self.aZ)-1, self.aZ[-1]+2, "{}%".format(self.aZ[-1]))

    #     self.figz.plot(self.gZ)
    #     self.figz.scatter(len(self.gZ)-1, self.gZ[-1])
    #     self.figz.text(len(self.gZ)-1, self.gZ[-1]+2, "{}%".format(self.gZ[-1]))

    #     self.figz.set_ylim(-25,25)

    #     plt.pause(0.001)


if __name__ == "__main__":
    robot_ids = rospy.get_param('/state_publisher/robot_id')

    # if not isinstance(robot_ids, list):
    #     robot_ids = [robot_ids]
    # laptop = Laptop(robot_ids, plot=False)
