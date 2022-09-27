#!/usr/bin/python3

import numpy as np
import rospy
import tf2_ros

from ar_track_alvar_msgs.msg import AlvarMarkers
from tf.transformations import euler_from_quaternion

class StateSubscriber:
    def __init__(self, base_id, **name_to_id):
        """
        Usage: robot_state = self.states[self.name_to_id["robot"]]
        """
        self.name_to_id = name_to_id
        self.id_to_name = {id:name for name, id in name_to_id.items()}

        # robot, object (optional), base, corner
        self.id_to_state = {id:np.zeros(3) for id in self.id_to_name}
        self.name_to_state = {name:self.id_to_state[id] for id, name in self.id_to_name.items()}
        self.base_id = base_id
        self.base_frame = f"ar_marker_{self.base_id}"
        self.n_full_updates = 0
        self.n_updates = 0

        print("setting up tf buffer/listener")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        print("finished setting up tf buffer/listener")

        print("waiting for /ar_pose_marker rostopic")
        rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.update, queue_size=1)
        print("subscribed to /ar_pose_marker")

    def update(self, msg):
        found_ids = [marker.id for marker in msg.markers]
        if not self.base_id in found_ids:
            print("\nBASE MARKER NOT FOUND\n")
            return

        updated_ids = {id:False for id in self.id_to_state}
        for marker in msg.markers:
            if marker.id in self.id_to_name and marker.id != self.base_id:
                state = self.id_to_state[marker.id]
                marker_frame = f"ar_marker_{marker.id}"
            else:
                continue

            for _ in range(100):
                try:
                    pose = self.tf_buffer.lookup_transform(self.base_frame, marker_frame, rospy.Time())
                    updated_ids[marker.id] = True
                    break
                except (tf2_ros.LookupException,
                        tf2_ros.ConnectivityException,
                        tf2_ros.ExtrapolationException):
                    pass

            t, r = pose.transform.translation, pose.transform.rotation
            quat = [r.x, r.y, r.z, r.w]
            roll, pitch, yaw = euler_from_quaternion(quat)

            state[0] = t.x
            state[1] = t.y
            state[2] = yaw

        if any(updated_ids.values()):
            self.n_updates += 1
        if all(updated_ids.values()):
            self.n_full_updates += 1

    def get_states(self, names, n_avg=2):
        if n_avg == 1:
            states = [self.name_to_state[name] for name in names]
        else:
            states = []
            while len(states) != n_avg:
                state = [self.name_to_state[name] for name in names]
                states.append(state)

                n_updates = self.n_updates
                time = rospy.get_time()
                while n_updates == self.n_updates:
                    if rospy.get_time() - time > 2:
                        print("STUCK IN STATE SUBSCRIBER get_states()")
                    rospy.sleep(0.0001)

            states = np.array(states).mean(axis=0)

        if len(names) > 1:
            return [state.squeeze() for state in states]
        else:
            return np.array(states).squeeze()
