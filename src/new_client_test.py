#!/usr/bin/env python3
import numpy as np
import rospy
from ros_stuff.msg import RobotCmd
from ros_stuff.srv import CommandAction  # Import service type
import sys


def laptop_client():
    # Initialize the client node
    rospy.init_node('laptop_client')

    serv_name1 = '/kami1/server'
    rospy.wait_for_service(serv_name1)
    serv_name2 = '/kami2/server'
    rospy.wait_for_service(serv_name2)
    try:
        # Acquire service proxy
        kami_proxy1 = rospy.ServiceProxy(
            serv_name1, CommandAction)
        rospy.loginfo('Command kami1')
        kami_proxy2 = rospy.ServiceProxy(
            serv_name2, CommandAction)
        rospy.loginfo('Command kami2')
        # Call cmd service via the proxy
        cmd = RobotCmd()
        cmd.left_pwm = 0.5
        cmd.right_pwm = 0.5
        kami_proxy1(cmd, 'kami1')
        kami_proxy2(cmd, 'kami2')
    except rospy.ServiceException as e:
        rospy.loginfo(e)


if __name__ == '__main__':
    laptop_client()