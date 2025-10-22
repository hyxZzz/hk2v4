import copy
import math as m
from typing import Dict

import numpy as np


actionDict: Dict[str, list] = {
    '0': [0, 1, 0, 0],  # 匀速前飞 不打导弹 0 1 2
    '1': [2, 1, 0, 0],  # 加速前飞 不打导弹 3 4 5
    '2': [4, 1, 0, 0],  # 加速前飞 不打导弹 6 7 8
    '3': [-2, 1, 0, 0],  # 减速前飞 9 10 11
    '4': [-4, 1, 0, 0],  # 减速前飞 12 13 14
    '5': [0, 2, 0, -1],  # 爬升 15 16 17
    '6': [0, 3, 0, -1],  # 爬升 18 19 20
    '7': [0, 4, 0, -1],  # 爬升 21 22 23
    '8': [0, 0, 0, -1],  # 俯冲 24 25 26
    '9': [0, -1, 0, -1],  # 俯冲 27 28 29
    '10': [0, -2, 0, -1],  # 俯冲 30 31 32
    '11': [0, 2, 0.25 * m.pi, -1],  # 左爬升 33 34 35
    '12': [0, 3, 0.25 * m.pi, -1],  # 左爬升 36 37 38
    '13': [0, 4, 0.25 * m.pi, -1],  # 左爬升 39 40 41
    '14': [0, 2, -0.25 * m.pi, -1],  # 右爬升 42 43 44
    '15': [0, 3, -0.25 * m.pi, -1],  # 右爬升 45 46 47
    '16': [0, 4, -0.25 * m.pi, -1],  # 右爬升 48 49 50
    '17': [0, -2, -0.25 * m.pi, -1],  # 左俯冲 51 52 53
    '18': [0, -3, -0.25 * m.pi, -1],  # 左俯冲 54 55 56
    '19': [0, -4, -0.25 * m.pi, -1],  # 左俯冲 57 58 59
    '20': [0, -2, 0.25 * m.pi, -1],  # 右俯冲 60 61 62
    '21': [0, -3, 0.25 * m.pi, -1],  # 右俯冲 63 64 65
    '22': [0, -4, 0.25 * m.pi, -1],  # 右俯冲 66 67 68
    '23': [0, 2, m.acos(1 / 2), 0],  # 左转弯 69 70 71
    '24': [0, 3, m.acos(1 / 3), 0],  # 左转弯 72 73 74
    '25': [0, 4, m.acos(1 / 4), 0],  # 左转弯 75 76 77
    '26': [0, 2, -m.acos(1 / 2), 0],  # 右转弯 78 79 80
    '27': [0, 3, -m.acos(1 / 3), 0],  # 右转弯 81 82 83
    '28': [0, 4, -m.acos(1 / 4), 0],  # 右转弯 84 85 86
}


def getActionDepository(missile_num: int, act_num: int = 29) -> np.ndarray:
    """生成包含导弹指派维度的动作库，第五列为目标编号（-1 表示不发射）。"""

    action_matrix = np.empty((act_num * (missile_num + 1), 5), dtype=np.float32)
    t = 0
    for i in range(act_num):
        for j in range(-1, missile_num):
            act = copy.deepcopy(actionDict[str(i)])
            act.append(j)
            action_matrix[t] = act
            t += 1
    return action_matrix


def getNewActionDepository(act_num: int = 29) -> np.ndarray:
    """保留旧接口，仅返回纯机动动作（无目标编号）。"""

    action_matrix = np.empty((act_num, 4), dtype=np.float32)
    for i in range(act_num):
        action_matrix[i] = actionDict[str(i)]
    return action_matrix
