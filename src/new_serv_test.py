#!/usr/bin/python3

import sys

import rospy
from ros_stuff.msg import MultiRobotCmd
from std_msgs.msg import Int8

from gpiozero import PWMOutputDevice, DigitalOutputDevice

# below are the correct ones for us
MOTOR_STANDBY = 17 # STBY - 11, GPIO 17
MOTOR_RIGHT_PWM = 18 # PWMB - 12, GPIO 18
MOTOR_RIGHT_FW = 23 # BIN1 - 16, GPIO 23
MOTOR_RIGHT_BW = 24 # BIN2 - 18, GPIO 24
MOTOR_LEFT_PWM = 13 # PWMA - 33, GPIO 13
MOTOR_LEFT_FW = 22 # AIN2 - 15, GPIO 22
MOTOR_LEFT_BW = 27 # AIN1 - 13, GPIO 27

ac_num = -1
def kami_callback(msg):
    print("Message received at time:", rospy.get_rostime())

    for cmd in msg.robot_commands:
        if cmd.robot_id == robot_id:
            left_action, right_action, duration = cmd.left_pwm, cmd.right_pwm, cmd.duration
            break

    if left_action > 0:
        motor_left_forward.on()
        motor_left_backward.off()
    else:
        motor_left_forward.off()
        motor_left_backward.on()

    if right_action > 0:
        motor_right_forward.on()
        motor_right_backward.off()
    else:
        motor_right_forward.off()
        motor_right_backward.on()

    motor_left_pwm.value = abs(left_action)
    motor_right_pwm.value = abs(right_action)

    action_receipt_msg = Int8()
    action_receipt_msg.data = robot_id
    publisher.publish(action_receipt_msg)

    print(f"{rospy.get_name()}  ||  L: {left_action}  ||  R: {right_action} || T: {duration}")

    rospy.sleep(duration)

    motor_left_pwm.off()
    motor_right_pwm.off()


if __name__ == '__main__':
    motor_standby = DigitalOutputDevice(MOTOR_STANDBY)
    motor_left_pwm = PWMOutputDevice(MOTOR_LEFT_PWM)
    motor_left_forward = DigitalOutputDevice(MOTOR_LEFT_FW)
    motor_left_backward = DigitalOutputDevice(MOTOR_LEFT_BW)
    motor_right_pwm = PWMOutputDevice(MOTOR_RIGHT_PWM)
    motor_right_forward = DigitalOutputDevice(MOTOR_RIGHT_FW)
    motor_right_backward = DigitalOutputDevice(MOTOR_RIGHT_BW)
    ports = [motor_standby, motor_left_pwm, motor_left_forward, motor_left_backward,
        motor_right_pwm, motor_right_forward, motor_right_backward]
    motor_standby.on()

    robot_name = sys.argv[1]
    robot_id = int(robot_name[-1])
    rospy.init_node(f'robot_{robot_name}')

    print("waiting for /action_topic rostopic")
    rospy.Subscriber("/action_topic", MultiRobotCmd, kami_callback, queue_size=1)
    print("subscribed to /action_topic")

    publisher = rospy.Publisher(f"/action_receipt_{robot_id}", Int8, queue_size=1)

    print("rospy spinning")
    rospy.spin()
