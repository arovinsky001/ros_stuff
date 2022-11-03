#!/usr/bin/python3

import rospy
from ros_stuff.msg import RobotCmd

from std_msgs.msg import Time
import sys

# from gpiozero import PWMOutputDevice, DigitalOutputDevice
from gpiozero import Motor, DigitalOutputDevice

# below are the correct ones for us
MOTOR_STANDBY = 17 # STBY - 11, GPIO 17
MOTOR_RIGHT_PWM = 18 # PWMB - 12, GPIO 18
MOTOR_RIGHT_FW = 23 # BIN1 - 16, GPIO 23
MOTOR_RIGHT_BW = 24 # BIN2 - 18, GPIO 24
MOTOR_LEFT_PWM = 13 # PWMA - 33, GPIO 13
MOTOR_LEFT_FW = 22 # AIN2 - 15, GPIO 22
MOTOR_LEFT_BW = 27 # AIN1 - 13, GPIO 27

def kami_callback(msg):
    print("Hi, I got a message:", msg)

    left_action, right_action, duration = msg.left_pwm, msg.right_pwm, msg.duration

    right_forward = right_action > 0

    # if left_forward:
    #     motor_left_forward.on()
    #     motor_left_backward.off()
    # else:
    #     motor_left_forward.off()
    #     motor_left_backward.on()
    # motor_left_pwm.on()

    # if right_forward:
    #     motor_right_forward.on()
    #     motor_right_backward.off()
    # else:
    #     motor_right_forward.off()
    #     motor_right_backward.on()
    # motor_right_pwm.on()

    left_pwm, right_pwm = abs(left_action), abs(right_action)

    if left_action > 0:
        left_motor.forward(left_pwm)
    else:
        left_motor.backward(left_pwm)

    if right_action > 0:
        right_motor.forward(right_pwm)
    else:
        right_motor.backward(right_pwm)

    time_msg = Time()
    time_msg.data = rospy.get_rostime()
    print(f"{rospy.get_name()}  ||  L: {left_pwm}  ||  R: {right_pwm} || T: {duration}")

    publisher.publish(time_msg)

    rospy.sleep(duration)

    left_motor.stop()
    right_motor.stop()


    # # take action
    # off_time = 0.2
    # motor_left_pwm.blink(on_time=duration, off_time=off_time, fade_in_time=0, fade_out_time=0, n=1, background=True)
    # motor_right_pwm.blink(on_time=duration, off_time=off_time, fade_in_time=0, fade_out_time=0, n=1, background=True)


if __name__ == '__main__':
    motor_standby = DigitalOutputDevice(MOTOR_STANDBY)
    # motor_left_pwm = PWMOutputDevice(MOTOR_LEFT_PWM)
    # motor_left_forward = DigitalOutputDevice(MOTOR_LEFT_FW)
    # motor_left_backward = DigitalOutputDevice(MOTOR_LEFT_BW)
    # motor_right_pwm = PWMOutputDevice(MOTOR_RIGHT_PWM)
    # motor_right_forward = DigitalOutputDevice(MOTOR_RIGHT_FW)
    # motor_right_backward = DigitalOutputDevice(MOTOR_RIGHT_BW)
    # ports = [motor_standby, motor_left_pwm, motor_left_forward, motor_left_backward,
    #     motor_right_pwm, motor_right_forward, motor_right_backward]
    motor_standby.on()

    left_motor = Motor(MOTOR_LEFT_FW, MOTOR_LEFT_BW, enable=MOTOR_STANDBY, pwm=True)
    right_motor = Motor(MOTOR_RIGHT_FW, MOTOR_RIGHT_BW, enable=MOTOR_STANDBY, pwm=True)

    rospy.init_node(f'robot_{sys.argv[1]}')

    print("waiting for /action_topic rostopic")
    rospy.Subscriber("/action_topic", RobotCmd, kami_callback, queue_size=1)
    print("subscribed to /action_topic")

    publisher = rospy.Publisher("/action_timestamps", Time, queue_size=1)

    print("rospy spinning")
    rospy.spin()
