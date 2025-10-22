import math as m
from typing import List, Tuple

import numpy as np

from Environment.env import ManeuverEnv
from flat_models.trajectory import Aircraft, Missiles
from utils.common import ComputeHeading, ComputePitch


def _generate_aircraft(num_planes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = []
    headings = []
    pitches = []
    speeds = []
    for _ in range(num_planes):
        x = np.random.uniform(-10000, 10000)
        y = np.random.uniform(-10000, 10000)
        z = np.random.uniform(8000, 12000)
        v = np.random.uniform(0.6, 1.2) * 340.0
        heading = np.random.uniform(-1, 1) * m.pi
        pitch = np.random.uniform(-5, 5) * m.pi / 180
        positions.append((x, z, y))
        headings.append(heading)
        pitches.append(pitch)
        speeds.append(v)
    return np.array(positions), np.array(headings), np.array(pitches), np.array(speeds)


def _generate_missiles(num_missiles: int, anchor: np.ndarray) -> Tuple[List[List[float]], List[float], List[float]]:
    center_x = np.mean(anchor[:, 0])
    center_y = np.mean(anchor[:, 2])
    center_z = np.mean(anchor[:, 1])
    angles = np.random.uniform(0, 2 * m.pi, size=num_missiles)
    radial_scale = np.random.uniform(0.85, 1.25, size=num_missiles)
    major_axis = 22000.0
    minor_axis = 16000.0
    missile_x = center_x + major_axis * radial_scale * np.cos(angles)
    missile_y = center_y + minor_axis * radial_scale * np.sin(angles)
    altitude_offsets = np.random.uniform(-3500.0, 3500.0, size=num_missiles)
    missile_z = np.clip(center_z + altitude_offsets, 0.0, None)
    positions = []
    headings = []
    pitches = []
    for i in range(num_missiles):
        pos = [missile_x[i], missile_z[i], missile_y[i]]
        positions.append(pos)
        headings.append(ComputeHeading([center_x, center_z, center_y], pos))
        pitches.append(ComputePitch([center_x, center_z, center_y], pos))
    return positions, headings, pitches


def init_env(
    num_missiles: int = 4,
    StepNum: int = 3500,
    interceptor_num: int = 12,
    num_planes: int = 2,
):
    aircraft_positions, headings, pitches, speeds = _generate_aircraft(num_planes)
    aircraft_list: List[Aircraft] = []
    for idx in range(num_planes):
        aircraft_list.append(
            Aircraft(
                list(aircraft_positions[idx]),
                V=float(speeds[idx]),
                Pitch=float(pitches[idx]),
                Heading=float(headings[idx]),
            )
        )

    missile_positions, missile_headings, missile_pitches = _generate_missiles(num_missiles, aircraft_positions)
    missile_speed = np.random.uniform(2.2, 3.0) * 340.0
    missiles_list = [
        Missiles(pos, V=missile_speed, Pitch=missile_pitches[i], Heading=missile_headings[i])
        for i, pos in enumerate(missile_positions)
    ]

    env = ManeuverEnv(
        missiles_list,
        aircraft_list,
        planeSpeed=list(speeds),
        missilesNum=num_missiles,
        spaceSize=StepNum,
        missilesSpeed=missile_speed,
        InterceptorNum=interceptor_num,
        allow_cross_lock=True,
        max_locks_per_missile=3,
    )

    return env, aircraft_list, missiles_list
