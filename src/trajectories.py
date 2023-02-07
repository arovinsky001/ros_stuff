#!/usr/bin/python3

import numpy as np


def figure8_trajectory(steps_per_episode):
    back_circle_center = np.array([1.4, 0.7])
    front_circle_center = np.array([0.8, 0.7])
    radius = np.linalg.norm(back_circle_center - front_circle_center) / 2

    goals = []

    for step in range(steps_per_episode):
        t_rel = step / steps_per_episode

        if t_rel < 0.5:
            theta = t_rel * 2 * 2 * np.pi
            center = front_circle_center
        else:
            theta = np.pi - ((t_rel - 0.5) * 2 * 2 * np.pi)
            center = back_circle_center

        goal = center + np.array([np.cos(theta), np.sin(theta)]) * radius
        goals.append(goal)

    return np.array(goals)

def S_trajectory(steps_per_episode):
    back_circle_center = np.array([1.4, 0.7])
    front_circle_center = np.array([0.8, 0.7])
    radius = np.linalg.norm(back_circle_center - front_circle_center) / 2

    goals = []

    for step in range(steps_per_episode):
        t_rel = step / steps_per_episode

        if t_rel < 0.5:
            theta = t_rel * 2 * np.pi + np.pi
            center = front_circle_center
        else:
            theta = np.pi - ((t_rel - 0.5) * 2 * np.pi)
            center = back_circle_center

        goal = center + np.array([np.cos(theta), np.sin(theta)]) * radius
        goals.append(goal)

    return np.array(goals)

def W_trajectory(steps_per_episode):
    points = np.array([
        [0.5, 0.9],
        [0.75, 0.3],
        [1.0, 0.6],
        [1.25, 0.3],
        [1.5, 0.9],
    ])

    distances_per_segment = np.linalg.norm(points[1:] - points[:-1], axis=1)
    distance_per_episode = np.sum(distances_per_segment)
    distance_per_step = distance_per_episode / steps_per_episode
    steps_per_segment = np.floor(distances_per_segment / distance_per_step).astype('int')
    new_steps_per_episode = steps_per_segment.sum()

    startpoints_x = points[:-1, 0]
    startpoints_y = points[:-1, 1]
    endpoints_x = points[1:, 0]
    endpoints_y = points[1:, 1]

    goals = np.empty((0, 2))

    for steps, startpoint, endpoint in zip(steps_per_segment, points[:-1], points[1:]):
        segment_points = np.linspace(startpoint, endpoint, steps)
        goals = np.concatenate((goals, segment_points), axis=0)

    return goals, new_steps_per_episode
