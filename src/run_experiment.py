#!/usr/bin/python3

import os
import argparse
import pickle as pkl
import numpy as np
import rospy
from tqdm import trange

from mpc_agent import MPCAgent
from replay_buffer import ReplayBuffer
from logger import Logger
from train_utils import AGENT_PATH, train_from_buffer
from data_utils import dimensions

from ros_stuff.msg import RobotCmd, ProcessedStates
from ros_stuff.srv import CommandAction

# seed for reproducibility
SEED = 0
import torch; torch.manual_seed(SEED)
np.random.seed(SEED)


class Experiment():
    def __init__(self, robot_pos, object_pos, corner_pos, robot_vel, object_vel, state_timestamp,
                 robot_id, object_id, mpc_horizon, mpc_samples, n_rollouts, tolerance, lap_time,
                 calibrate, plot, new_buffer, pretrain, robot_goals, scale, mpc_method, save_freq,
                 online, mpc_refine_iters, pretrain_samples, consecutive, random_steps, rate, use_all_data, debug,
                 save_agent, load_agent, train_epochs, mpc_gamma, ensemble, batch_size, rand_ac_mean,
                 meta, **kwargs):
        # flags for different stages of eval
        self.started = False
        self.done = False

        # counters
        self.steps = 0
        self.time_elapsed = 0.
        self.laps = 0

        # AR tag ids for state lookup
        self.robot_id = robot_id
        self.object_id = object_id

        # states
        self.robot_pos = robot_pos
        self.object_pos = object_pos
        self.corner_pos = corner_pos
        self.robot_vel = robot_vel
        self.object_vel = object_vel
        self.state_timestamp = state_timestamp

        # online data collection/learning params
        self.random_steps = 0 if pretrain or load_agent else random_steps
        self.gradient_steps = 3
        self.online = online
        self.save_freq = save_freq
        self.pretrain_samples = pretrain_samples

        # system params
        self.rate = rate
        self.debug = debug
        self.use_object = (self.object_id >= 0)
        self.duration = 0.4 if self.use_object else 0.2
        self.action_range = np.array([[-1, -1], [1, 1]]) * 0.999
        self.rand_ac_mean = rand_ac_mean
        self.post_action_sleep_time = 0.3

        # train params
        self.pretrain = pretrain
        self.consecutive = consecutive
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.use_all_data = use_all_data
        self.save_agent = save_agent
        self.load_agent = load_agent

        # misc
        self.n_rollouts = n_rollouts
        self.tolerance = tolerance
        self.lap_time = lap_time
        self.robot_goals = robot_goals
        self.scale = scale
        self.debug = debug
        self.meta = meta
        self.start_lap_time = np.random.rand() * lap_time
        self.reverse_lap = False

        if not robot_goals:
            assert self.use_object

        self.all_actions = []
        self.costs = np.empty((0, 4))      # dist, heading, perp, total

        self.logger = Logger(self, plot, corner_pos, **kwargs)
        self.replay_buffer, self.validation_buffer = self.logger.load_buffer(robot_id)

        if new_buffer or self.replay_buffer is None:
            state_dim = 2 * dimensions["state_dim"] if self.use_object else dimensions["state_dim"]
            self.replay_buffer = ReplayBuffer(capacity=100000, state_dim=state_dim, action_dim=dimensions["action_dim"])

        if not self.debug:
            print(f"waiting for robot {self.robot_id} service")
            rospy.wait_for_service(f"/kami{self.robot_id}/server")
            self.service_proxy = rospy.ServiceProxy(f"/kami{self.robot_id}/server", CommandAction)
            print("connected to robot service")

        self.yaw_offset_path = "/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/sim/data/yaw_offsets.npy"
        if not os.path.exists(self.yaw_offset_path) or calibrate:
            self.yaw_offsets = np.zeros(10)
            self.calibrate()

        self.yaw_offsets = np.load(self.yaw_offset_path)

        self.define_goal_trajectory()

        # weights for MPC cost terms
        if self.robot_goals:
            self.cost_weights = {
                "distance": 1.,
                "heading": 0.,
                "perpendicular": 0.,
                "action_norm": 0.,
                "distance_bonus": 0.,
                "separation": 0.,
                "heading_difference": 0.,
            }
        else:
            self.cost_weights = None
            # self.cost_weights = {
            #     "distance": 1.,
            #     "heading": 0.1,
            #     "perpendicular": 0.,
            #     "action_norm": 0.,
            #     "distance_bonus": 0.,
            #     "separation": 0.1,
            #     "heading_difference": 0.1,
            # }

        # parameters for MPC methods
        self.mpc_params = {
            "horizon": mpc_horizon,
            "sample_trajectories": mpc_samples,
            "robot_goals": robot_goals,
        }
        if mpc_method == 'mppi':
            self.mpc_params.update({
                "beta": 0.5,
                "gamma": mpc_gamma,
                "noise_std": 2.,
            })
        elif mpc_method == 'cem':
            self.mpc_params.update({
                "alpha": 0.8,
                "n_best": 30,
                "refine_iters": mpc_refine_iters,
            })
        elif mpc_method == 'shooting':
            self.mpc_params.update({})
        else:
            raise NotImplementedError

        if load_agent:
            with open(AGENT_PATH, "rb") as f:
                self.agent = pkl.load(f)
        else:
            self.agent = MPCAgent(seed=SEED, mpc_method=mpc_method, dist=True, scale=self.scale,
                                  hidden_dim=200, hidden_depth=1, lr=0.001, std=0.1,
                                  ensemble=ensemble, use_object=self.use_object,
                                  action_range=self.action_range)

            if pretrain:
                train_from_buffer(
                    self.agent, self.replay_buffer, validation_buffer=self.validation_buffer,
                    pretrain=pretrain, pretrain_samples=pretrain_samples, consecutive=consecutive, save_agent=save_agent,
                    train_epochs=train_epochs, use_all_data=use_all_data, batch_size=batch_size,
                    meta=meta,
                )

        np.set_printoptions(suppress=True)

    def run(self):
        self.take_warmup_steps()

        r = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            t = rospy.get_time()

            if not self.started and self.dist_to_start() < self.tolerance \
                        and (self.pretrain or self.replay_buffer.size >= self.random_steps):
                self.started = True
                self.prev_goal = self.get_goal()

            state, action = self.step()
            self.logger.log_states(self.robot_pos, self.object_pos, self.prev_goal, self.started)

            if self.done:
                rospy.signal_shutdown("Finished! All robots reached goal.")
                return

            r.sleep()
            print("TIME:", rospy.get_time() - t)

            if not self.debug:
                next_state = self.get_state()
                self.collect_training_data(state, action, next_state)

            if self.online:
                self.update_model_online()

            if self.started:
                self.all_actions.append(action.tolist())

    def step(self):
        if not self.pretrain and not self.load_agent and self.replay_buffer.size == self.random_steps:
            train_from_buffer(
                self.agent, self.replay_buffer, validation_buffer=self.validation_buffer,
                save_agent=self.save_agent, train_epochs=self.train_epochs,
                use_all_data=self.use_all_data, batch_size=self.batch_size,
                meta=self.meta,
            )

        state = self.get_state()
        action = self.get_take_action(state)

        self.check_rollout_finished()

        self.steps += 1
        return state, action

    def take_warmup_steps(self):
        if self.debug:
            return

        for _ in trange(30, desc="Warmup Steps"):
            idx = np.random.randint(0, 2)
            negate = np.random.randint(0, 2)
            actions = np.array([[1, 1], [1, -1]]) * 0.999
            action = actions[idx] * (-1 if negate else 1)

            action_req = RobotCmd()
            action_req.left_pwm = action[0]
            action_req.right_pwm = action[1]
            action_req.duration = self.duration

            self.service_proxy(action_req, f"kami{self.robot_id}")
            rospy.sleep(self.post_action_sleep_time)

    def get_state(self):
        if np.any(self.robot_pos[:2] > self.corner_pos[:2]) or np.any(self.robot_pos[:2] < 0):
            print("\nOUT OF BOUNDS\n")
            import pdb;pdb.set_trace()

        robot_pos = self.robot_pos.copy()
        robot_pos[2] = (robot_pos[2] + self.yaw_offsets[self.robot_id]) % (2 * np.pi)

        if self.use_object:
            object_pos = self.object_pos.copy()
            object_pos[2] = (object_pos[2] + self.yaw_offsets[self.object_id]) % (2 * np.pi)

            return np.concatenate((robot_pos, object_pos), axis=0)
        else:
            return robot_pos

    def get_take_action(self, state):
        goal = self.get_goal()
        state_for_prev_goal = state[:dimensions["state_dim"]] if self.robot_goals else state[dimensions["state_dim"]:]
        prev_goal = self.prev_goal if self.started else state_for_prev_goal
        prev_goal = state_for_prev_goal

        if self.replay_buffer.size >= self.random_steps:
            action = self.agent.get_action(state, prev_goal, goal, cost_weights=self.cost_weights, params=self.mpc_params)
            self.time_elapsed += self.duration * (-1 if self.reverse_lap else 1) if self.started else 0
        else:
            print("TAKING RANDOM ACTION")
            # idx = self.replay_buffer.size % 2
            # locs = np.array([[1, 1], [1, -1]]) * self.rand_ac_mean
            # if self.replay_buffer.size < self.random_steps / 2:
            #     locs *= -1
            # scale = 0.7 if self.rand_ac_mean == 0 else 0.3
            # # action = np.random.normal(loc=locs[idx], scale=scale, size=dimensions["action_dim"])
            # action = np.random.uniform(*self.action_range, size=dimensions["action_dim"]).squeeze()

            rng = np.linspace(-1, 1, 3)
            left, right = np.meshgrid(rng, rng)
            actions = np.stack((left, right)).transpose(2, 1, 0).reshape(-1, 2)
            action = actions[self.replay_buffer.size]

        action = np.clip(action, *self.action_range)
        action_req = RobotCmd()
        action_req.left_pwm = action[0]
        action_req.right_pwm = action[1]
        action_req.duration = self.duration

        if not self.debug:
            print("SENDING ACTION")
            self.service_proxy(action_req, f"kami{self.robot_id}")
            rospy.sleep(self.post_action_sleep_time)

        print(f"\n\n\n\nNO. {self.steps}")
        print("/////////////////////////////////////////////////")
        print("=================================================")
        print("GOAL:", goal)
        print("STATE:", state)
        print("ACTION:", action)
        print("ACTION NORM:", np.linalg.norm(action) / np.sqrt(2), "\n")

        cost_dict, total_cost = self.record_costs(prev_goal, goal)

        for cost_type, cost in cost_dict.items():
            print(f"{cost_type}: {cost}")
        print("TOTAL:", total_cost)

        print("\nREPLAY BUFFER SIZE:", self.replay_buffer.size)
        print("=================================================")
        print("/////////////////////////////////////////////////")

        self.prev_goal = goal.copy()

        return action

    def calibrate(self):
        yaw_offsets = np.zeros(10)

        input(f"Place robot/object on the left calibration point, aligned with the calibration line and hit enter.")
        left_state = self.get_state()
        input(f"Place robot/object on the right calibration point, aligned with the calibration line and hit enter.")
        right_state = self.get_state()

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

    def define_goal_trajectory(self):
        rospy.sleep(0.2)        # wait for states to be published and set

        back_circle_center_rel = np.array([0.7, 0.5])
        front_circle_center_rel = np.array([0.4, 0.5])

        self.back_circle_center = back_circle_center_rel * self.corner_pos[:2]
        self.front_circle_center = front_circle_center_rel * self.corner_pos[:2]
        self.radius = np.linalg.norm(self.back_circle_center - self.front_circle_center) / 2

    def record_costs(self, prev_goal, goal):
        cost_dict = self.agent.compute_costs(
                self.get_state()[None, None, None, :], np.array([[[[0., 0.]]]]), prev_goal, goal, robot_goals=self.robot_goals
                )

        total_cost = 0
        for cost_type, cost in cost_dict.items():
            cost_dict[cost_type] = cost.squeeze()
            if cost_type != "distance":
                total_cost += cost.squeeze() * self.cost_weights[cost_type]
        total_cost = (total_cost + self.cost_weights["distance"]) * cost_dict["distance"]

        if self.started:
            costs_to_record = np.array([[cost_dict["distance"], cost_dict["heading"], cost_dict["perpendicular"], total_cost]])
            self.costs = np.append(self.costs, costs_to_record, axis=0)

        return cost_dict, total_cost

    def dist_to_start(self):
        state = self.get_state().squeeze()
        state = state[:3] if self.robot_goals else state[3:]
        return np.linalg.norm((state - self.get_goal(time_override=0.))[:2])

    def get_goal(self, time_override=None):
        if time_override is not None:
            time_elapsed = time_override
        else:
            time_elapsed = self.time_elapsed

        t_rel = ((time_elapsed + self.start_lap_time) % self.lap_time) / self.lap_time

        if t_rel < 0.5:
            theta = t_rel * 2 * 2 * np.pi
            center = self.front_circle_center
        else:
            theta = np.pi - ((t_rel - 0.5) * 2 * 2 * np.pi)
            center = self.back_circle_center

        goal = center + np.array([np.cos(theta), np.sin(theta)]) * self.radius
        return np.block([goal, 0.0])

    def check_rollout_finished(self):
        if np.abs(self.time_elapsed) > self.lap_time:
            self.laps += 1
            # Print current cumulative loss per lap completed
            dist_costs, heading_costs, perp_costs, total_costs = self.costs.T
            data = np.array([[dist_costs.mean(), dist_costs.std(), dist_costs.min(), dist_costs.max()],
                         [perp_costs.mean(), perp_costs.std(), perp_costs.min(), perp_costs.max()],
                         [heading_costs.mean(), heading_costs.std(), heading_costs.min(), heading_costs.max()],
                         [total_costs.mean(), total_costs.std(), total_costs.min(), total_costs.max()]])
            print("lap:", self.laps)
            print("rows: (dist, perp, heading, total)")
            print("cols: (mean, std, min, max)")
            print("DATA:", data, "\n")
            self.time_elapsed = 0.
            self.started = False

            start_state = self.get_goal(time_override=0.)
            self.logger.plot_states(save=True, laps=self.laps, replay_buffer=self.replay_buffer,
                                    start_state=start_state, reverse_lap=self.reverse_lap)
            self.logger.reset_plot_states()
            self.logger.log_performance_metrics(self.costs, self.all_actions)

            if self.online:
                self.logger.dump_agent(self.agent, self.laps, self.replay_buffer)
                self.costs = np.empty((0, 4))
            elif(not self.online and self.laps == self.n_rollouts):
                self.logger.dump_agent(self.agent, self.laps, self.replay_buffer)

            self.start_lap_time = np.random.rand() * self.lap_time
            self.reverse_lap = not self.reverse_lap

        if self.laps == self.n_rollouts:
            self.done = True

    def collect_training_data(self, state, action, next_state):
        self.replay_buffer.add(state, action, next_state)

        if self.replay_buffer.idx % self.save_freq == 0:
            print(f"\nSAVING REPLAY BUFFER\n")
            self.logger.dump_buffer(self.replay_buffer)

    def update_model_online(self):
        if self.replay_buffer.size >= self.random_steps:
            for model in self.agent.models:
                for _ in range(self.gradient_steps):
                    states, actions, next_states = self.replay_buffer.sample(self.batch_size)
                    model.update(states, actions, next_states)

def main(args):
    rospy.init_node("laptop_client_mpc")

    pos_dim = 3
    vel_dim = 3
    robot_pos = np.empty(pos_dim)
    object_pos = np.empty(pos_dim)
    corner_pos = np.empty(pos_dim)
    robot_vel = np.empty(vel_dim)
    object_vel = np.empty(vel_dim)
    state_timestamp = np.empty(1)

    def update_state(msg):
        rs, os, cs = msg.robot_state, msg.object_state, msg.corner_state

        robot_pos[:] = np.array([rs.x, rs.y, rs.yaw])
        object_pos[:] = np.array([os.x, os.y, os.yaw])
        corner_pos[:] = np.array([cs.x, cs.y, cs.yaw])

        robot_vel[:] = np.array([rs.x_vel, rs.y_vel, rs.yaw_vel])
        object_vel[:] = np.array([os.x_vel, os.y_vel, os.yaw_vel])

        secs, nsecs = msg.header.stamp.secs, msg.header.stamp.nsecs
        state_timestamp[:] = secs + nsecs / 1e9

    print("waiting for /processed_state topic from state publisher")
    rospy.Subscriber("/processed_state", ProcessedStates, update_state, queue_size=1)
    print("subscribed to /processed_state")

    experiment = Experiment(robot_pos, object_pos, corner_pos, robot_vel, object_vel, state_timestamp, **vars(args))
    experiment.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do MPC.')
    parser.add_argument('-robot_id', type=int, help='robot id for rollout')
    parser.add_argument('-object_id', type=int, default=-1, help='object id for rollout')
    parser.add_argument('-mpc_method', default='mppi')
    parser.add_argument('-mpc_horizon', type=int)
    parser.add_argument('-mpc_samples', type=int)
    parser.add_argument('-mpc_refine_iters', type=int, default=1)
    parser.add_argument('-mpc_gamma', type=float, default=50000)
    parser.add_argument('-n_rollouts', type=int)
    parser.add_argument('-tolerance', type=float, default=0.05)
    parser.add_argument('-lap_time', type=float)
    parser.add_argument('-calibrate', action='store_true')
    parser.add_argument('-plot', action='store_true')
    parser.add_argument('-new_buffer', action='store_true')
    parser.add_argument('-pretrain', action='store_true')
    parser.add_argument('-consecutive', action='store_true')
    parser.add_argument('-robot_goals', action='store_true')
    parser.add_argument('-scale', action='store_true')
    parser.add_argument('-online', action='store_true')
    parser.add_argument('-save_freq', type=int, default=50)
    parser.add_argument('-pretrain_samples', type=int, default=500)
    parser.add_argument('-random_steps', type=int, default=500)
    parser.add_argument('-rate', type=float, default=1.)
    parser.add_argument('-use_all_data', action='store_true')
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-save_agent', action='store_true')
    parser.add_argument('-load_agent', action='store_true')
    parser.add_argument('-train_epochs', type=int, default=200)
    parser.add_argument('-ensemble', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=1000)
    parser.add_argument('-rand_ac_mean', type=float, default=0.)
    parser.add_argument('-rand_rot', action='store_true')
    parser.add_argument('-exp_name')
    parser.add_argument('-meta', action='store_true')


    args = parser.parse_args()
    main(args)
