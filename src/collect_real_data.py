#!/usr/bin/python3

import argparse
import numpy as np
import rospy

from ros_stuff.msg import RobotCmd
from ros_stuff.src.utils import KamigamiInterface


SAVE_PATH = "/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/sim/data/real_data.npz"

class DataCollector(KamigamiInterface):
    def __init__(self, robot_ids):
        super().__init__(robot_ids, SAVE_PATH)

    def run(self):
        while not rospy.is_shutdown():
            self.step()
            if self.done:
                rospy.signal_shutdown("finished collecting training data!")
                return

    def step(self):
        if self.n_updates == 0:
            return

        states = self.get_states(perturb=True)
        if not states:
            return

        actions = self.get_take_actions()

        next_states = self.get_states(perturb=True)
        if not next_states:
            return

        print(f"\nstates:, {self.current_states}")
        print(f"actions: {actions}")
        
        self.states.append(states)
        self.actions.append(actions)
        self.next_states.append(next_states)
        
        if len(self.states) % self.save_freq == 0:
            self.save_training_data()

    def get_take_actions(self):
        actions = np.random.uniform(*self.action_range, size=(len(self.robot_ids), self.action_range.shape[-1]))
        actions = np.append(actions, np.empty((len(self.robot_ids), 1)), axis=1)

        reqs = [RobotCmd() for _ in range(len(self.robot_ids))]
        for i, req in enumerate(reqs):
            req.left_pwm = actions[i, 0]
            req.right_pwm = actions[i, 1]
            req.duration = actions[i, 2]
            actions[i, 3] = self.robot_ids[i]
        
        for i, proxy in enumerate(self.service_proxies):
            proxy(reqs[i], f'kami{self.robot_ids[i]}')

        n_updates = self.n_updates
        time = rospy.get_time()
        while self.n_updates - n_updates < self.n_wait_updates:
            if rospy.get_time() - time > 5:
                print("DIDN'T UPDATE FOR TOO LONG")
                import pdb;pdb.set_trace()
            rospy.sleep(0.001)
            i += 1

        return actions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect random training data.')
    parser.add_argument('-robot_ids', type=int, default=np.array([0, 2]), help='robot id for rollout')

    args = parser.parse_args()

    dc = DataCollector(args.robot_ids)
    dc.run()
