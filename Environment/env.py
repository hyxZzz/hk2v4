from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import math as m
import numpy as np
from gym import spaces

from Environment.ActionDepository import getNewActionDepository
from Environment.reset_env import reset_para
from flat_models.ThreatEvaluate import CalTreat
from flat_models.trajectory import Aircraft, Interceptor, Missiles
from utils.common import CalDistance

act_num = 29  # 机动动作的多少
Gostep = 1  # 机动策略改变的频率

INTERCEPT_SUCCESS_DISTANCE = 20.0  # 拦截弹与导弹的命中阈值
MISSILE_HIT_DISTANCE = 20.0  # 来袭导弹命中飞机的阈值
DANGER_DISTANCE = 3000  # 危险距离 用于奖励函数的非线性分段
LanchGap = 70  # 发射间隔
ERRACTIONSCALE = 2  # 惩罚加的系数 原先设为10
DANGERSCALE = 3  # 危险情况下 距离影响的系数 原先设为5


class TerminationStatus(Enum):
    RUNNING = -1
    FAIL_MISSILE_HIT = 0
    MANEUVER_SUCCESS = 1
    INTERCEPT_SUCCESS = 2
    FAIL_CRASH = 3
    TIMEOUT = 4


@dataclass
class RewardWeights:
    height: float = 0.6
    distance: float = 0.8
    interceptor_usage: float = 0.4
    treat: float = 0.4
    intercept_quality: float = 0.5
    sparse: float = 0.75


class ManeuverEnv:
    """
                        导弹编号	    X位置	Y位置	Z位置	速度	    俯仰角	偏转角
                            飞机  	X位置	Y位置	Z位置	速度	    俯仰角	偏转角
                        拦截弹编号    X位置	Y位置	Z位置	速度	    俯仰角	偏转角
                        """

    def __init__(
        self,
        missileList: List[Missiles],
        aircraftList: Aircraft,
        planeSpeed: float = 170.0,
        missilesNum: int = 3,
        spaceSize: int = 5000,
        missilesSpeed: float = 680.0,
        InterceptorNum: int = 8,
        InterceptorSpeed: float = 540.0,
        curiosity_scale: float = 0.5,
        punish_penalty: float = 1.0,
        launch_gap: int = LanchGap,
        reward_weights: RewardWeights | None = None,
    ) -> None:
        self.reward_weights = reward_weights or RewardWeights()
        self.curiosity_scale = curiosity_scale
        self.punish_penalty = punish_penalty
        self.launch_gap = max(1, int(launch_gap))

        self.escapeFlag: TerminationStatus = TerminationStatus.RUNNING
        self.status_messages = {
            TerminationStatus.RUNNING: 'Go on combating...'.capitalize(),
            TerminationStatus.FAIL_MISSILE_HIT: 'Hit confirmed, escape failed.',
            TerminationStatus.MANEUVER_SUCCESS: 'Maneuver success.',
            TerminationStatus.INTERCEPT_SUCCESS: 'Intercept success.',
            TerminationStatus.FAIL_CRASH: 'Vehicle lost due to crash.',
            TerminationStatus.TIMEOUT: 'Mission timeout.',
        }

        self.missileNum = missilesNum
        self.missileSpeed = missilesSpeed
        self.planeSpeed = planeSpeed
        self.interceptorNum = InterceptorNum
        self.interceptorSpeed = InterceptorSpeed

        self.action_dep = getNewActionDepository(act_num)
        self.maneuver_action_count = self.action_dep.shape[0]
        self.target_bucket = self.missileNum + 1  # -1 表示不发射
        self.action_space = spaces.Discrete(self.maneuver_action_count * self.target_bucket)

        self.spaceSize = spaceSize
        self.max_steps = max(1, int(spaceSize))
        self.Treat_t = 0.0

        self.missileList = missileList
        self.aircraftList = aircraftList
        self.interceptorList: List[Interceptor] = []
        self.interceptor_remain = InterceptorNum
        self.position_scale = 25000.0
        self.At_1 = 0

        for _ in range(InterceptorNum):
            self.interceptorList.append(
                Interceptor(
                    [self.aircraftList.X, self.aircraftList.Y, self.aircraftList.Z],
                    self.aircraftList.V,
                    self.aircraftList.Pitch,
                    self.aircraftList.Heading,
                )
            )

        self.observation_planes = np.zeros((1, 6), dtype=np.float32)
        self.observation_missiles = np.zeros((self.missileNum, 6), dtype=np.float32)
        self.observation_interceptors = np.zeros((self.interceptorNum, 6), dtype=np.float32)
        self.StateShape = (
            self.observation_planes.shape[0]
            + self.observation_missiles.shape[0]
            + self.observation_interceptors.shape[0]
        )

        self.D0 = np.empty((missilesNum,), dtype=np.float32)
        for i in range(missilesNum):
            self.D0[i] = CalDistance(
                [self.aircraftList.X, self.aircraftList.Y, self.aircraftList.Z],
                [self.missileList[i].X, self.missileList[i].Y, self.missileList[i].Z],
            )
        self.t = 0
        self.lanchTime = 0

    """飞机动作库，11种机动动作，利用三个控制量来控制。

            输入参数：

            ----------

            commandNum:动作指令序号

            Returns

            -----------

            控制量nx,nz,roll和俯仰角限制标识                        
        """

    def AirCraftActions(self, commandNum: int, interceptor_goal: int) -> Tuple[np.ndarray, int]:
        commandNum = int(np.clip(commandNum, 0, self.maneuver_action_count - 1))
        action = self.action_dep[commandNum][0:4]
        launch_successful = self.LanchPolicy(interceptor_goal)
        actual_goal = interceptor_goal if launch_successful else -1
        return action, actual_goal

    def decode_action(self, action: int) -> Tuple[int, int]:
        action = int(np.clip(action, 0, self.action_space.n - 1))
        maneuver_idx = action // self.target_bucket
        interceptor_bucket = action % self.target_bucket
        interceptor_goal = interceptor_bucket - 1
        return maneuver_idx, interceptor_goal

    def _set_status(self, status: TerminationStatus) -> None:
        self.escapeFlag = status

    def _build_info(self) -> dict:
        return {
            'status': self.escapeFlag.name,
            'code': self.escapeFlag.value,
            'message': self.status_messages[self.escapeFlag],
        }

    def _advance_environment(self, nx: float, ny: float, roll: float, pitch: float) -> np.ndarray:
        self._set_status(TerminationStatus.RUNNING)

        if self.t >= self.max_steps:
            self._set_status(TerminationStatus.TIMEOUT)
            return self._get_obs()

        tx, ty, tz = self.aircraftList.AircraftPostition(None, nx, ny, roll, pitch)
        self.observation_planes[0] = np.array(
            [tx, ty, tz, self.aircraftList.Pitch, self.aircraftList.Heading, self.aircraftList.roll]
        )

        x_a, y_a, z_a = self.aircraftList.X, self.aircraftList.Y, self.aircraftList.Z
        v_a, Pitch_a, Heading_a = self.aircraftList.V, self.aircraftList.Pitch, self.aircraftList.Heading
        ac_list = [x_a, y_a, z_a]

        active_missiles = 0
        for missile_index, missile in enumerate(self.missileList):
            if not missile.attacking:
                self.observation_missiles[missile_index] = np.array(
                    [missile.X, missile.Y, missile.Z, missile.Pitch, missile.Heading, 0]
                )
                continue

            active_missiles += 1
            mx, my, mz = missile.MissilePosition(ac_list, v_a, Pitch_a, Heading_a)
            self.observation_missiles[missile_index] = np.array(
                [mx, my, mz, missile.Pitch, missile.Heading, 0]
            )
            dist = m.sqrt((x_a - mx) ** 2 + (y_a - my) ** 2 + (z_a - mz) ** 2)

            for itr_index, interceptor in enumerate(self.interceptorList):
                if interceptor.T_i == -1:
                    interceptor.sync_with_aircraft(
                        [x_a, y_a, z_a], Pitch_a, Heading_a, self.aircraftList.V
                    )
                    self.observation_interceptors[itr_index] = np.array(
                        [x_a, y_a, z_a, Pitch_a, Heading_a, 0]
                    )
                    continue

                if interceptor.attacking == 1:
                    continue

                ix, iy, iz = interceptor.X_i, interceptor.Y_i, interceptor.Z_i
                if interceptor.T_i == missile_index:
                    ix, iy, iz = interceptor.InterceptorPosition(
                        [mx, my, mz], missile.V, missile.Pitch, missile.Heading
                    )
                    dist_im = m.sqrt((ix - mx) ** 2 + (iy - my) ** 2 + (iz - mz) ** 2)
                    if dist_im < INTERCEPT_SUCCESS_DISTANCE:
                        missile.attacking = False
                        interceptor.attacking = 1
                self.observation_interceptors[itr_index] = np.array(
                    [ix, iy, iz, interceptor.Pitch_i, interceptor.Heading_i, 0]
                )

            if dist < MISSILE_HIT_DISTANCE:
                self._set_status(TerminationStatus.FAIL_MISSILE_HIT)
                return self._get_obs()

        if active_missiles == 0:
            self._set_status(TerminationStatus.INTERCEPT_SUCCESS)

        return self._get_obs()

    def constraint_obs(self, act: List, speedFlag):
        nx, ny, roll, pitch = act
        if speedFlag:
            nx = m.sin(self.aircraftList.Pitch)
        else:
            ny = m.cos(self.aircraftList.Pitch) / max(m.cos(self.aircraftList.roll), 1e-6)
            roll = self.aircraftList.roll
            pitch = -1

        obs = self._advance_environment(nx, ny, roll, pitch)
        return obs, self.escapeFlag.value, self._build_info()

    def generate_obs(self, act: List):
        nx, ny, roll, pitch = act
        obs = self._advance_environment(nx, ny, roll, pitch)
        return obs, self.escapeFlag.value, self._build_info()

    """
            动作生成函数，输入机动策略序号与拦截目标导弹序号

            ---------------
            Returns

            机动动作列表[nx, ny, roll, Pitch]，并改变了类内interceptors的attacking标志和导引目标T_i
        """

    def _gen_action(self, c, goal=None):
        if goal == None:
            return

        return self.AirCraftActions(c, goal)

    def _get_obs(self):
        obs = np.concatenate((self.observation_planes, self.observation_missiles, self.observation_interceptors))
        return obs

    def _get_actSpace(self):
        a = self.action_space.n
        return a

    def _get_stateSpace(self):
        s = self._get_obs()
        s = s.flatten()
        return s.shape

    # 获取拉直的向量
    def _get_flattenstate(self, s):
        s = s.flatten()
        return s

    # 将拉直的向量重构为环境内计算的向量格式
    def _resize_flattenState(self, flattenState):
        obs = np.resize(flattenState, (self.StateShape, 6))
        return obs

    def render(self):
        pass


    """
        拦截弹锁定目标的上限
    """
    def LockConstraint(self, intceptor_goal):
        if intceptor_goal < 0:
            return False

        locked_Num = 0  # 打击第i个导弹的拦截弹个数
        for j in range(len(self.interceptorList)):
            Att_Num = self.interceptorList[j].T_i  # 拦截弹的锁定目标
            # 当拦截弹锁定目标为第i个导弹时，给变量+1
            if Att_Num == intceptor_goal:
                locked_Num += 1

        active_missiles = self.getRemainMissileNum()
        if active_missiles <= 0:
            return False

        base_limit = max(1, m.ceil(self.interceptorNum / max(1, self.missileNum)))
        focus_bonus = max(0, self.missileNum - active_missiles)
        max_lock = min(self.interceptorNum, base_limit + focus_bonus)

        if locked_Num >= max_lock:
            return False
        return True

    """ 
        飞机高度奖励：
        输入：飞机高度
        输出：奖励值：[0,1]
    
        """

    def heightReward(self, h: float) -> float:
        safe_min = 8000.0
        safe_max = 12000.0
        tolerance = 1000.0
        hard_min = safe_min - tolerance
        hard_max = safe_max + tolerance

        if h < hard_min or h > hard_max:
            self._set_status(TerminationStatus.FAIL_CRASH)
            return -1.0

        if h < safe_min:
            ratio = (h - hard_min) / max(safe_min - hard_min, 1.0)
            return -1.0 + 2.0 * np.clip(ratio, 0.0, 1.0)
        if h > safe_max:
            ratio = (hard_max - h) / max(hard_max - safe_max, 1.0)
            return -1.0 + 2.0 * np.clip(ratio, 0.0, 1.0)

        center = (safe_min + safe_max) / 2.0
        span = max((safe_max - safe_min) / 2.0, 1.0)
        offset = (h - center) / span
        return float(np.clip(1.0 - offset ** 2, -1.0, 1.0))

    """
        距离奖励：
        输入：导弹向量
        输出：奖励值【-1.608， 1】
    """
    def distanceReward(self, missileState: np.ndarray, planeState: np.ndarray) -> float:
        min_reward = 0.0
        has_active = False
        for idx in range(missileState.shape[0]):
            if not self.missileList[idx].attacking:
                continue
            has_active = True
            distance = float(np.linalg.norm(missileState[idx][:3] - planeState[:3]))
            normalized = (distance - DANGER_DISTANCE) / max(DANGER_DISTANCE, 1.0)
            reward = float(np.clip(normalized, -1.0, 1.0))
            if idx == 0 or reward < min_reward:
                min_reward = reward

        return min_reward if has_active else 0.0




    """
        成型奖励：输入全局状态，输出成型奖励值

        ---------------
        Returns

        成型奖励值
    """

    def commonReward(self, state: np.ndarray, actual_goal: int, desired_goal: int) -> float:
        plane_state = state[0]
        missile_state = state[1 : self.missileNum + 1]

        closest_dist, threat_index = self.getClosetMissileDist()
        danger_flag = closest_dist <= DANGER_DISTANCE

        height_component = self.heightReward(float(plane_state[1]))
        distance_component = self.distanceReward(missile_state, plane_state)

        treat_value = 0.0
        v_p = self.aircraftList.V
        for idx in range(missile_state.shape[0]):
            if not self.missileList[idx].attacking:
                continue
            missile_pos = missile_state[idx]
            v_m = self.missileList[idx].V
            treat_value = max(treat_value, CalTreat(plane_state, missile_pos, v_p, v_m))
        treat_component = float(np.clip(treat_value / 10.0, -1.0, 1.0))

        interceptor_component = 0.0
        if actual_goal == threat_index and actual_goal != -1:
            interceptor_component = 1.0
        elif actual_goal == -1 and danger_flag:
            interceptor_component = -1.0
        elif actual_goal not in (-1, threat_index):
            interceptor_component = -0.6
        elif desired_goal != actual_goal and desired_goal >= 0:
            interceptor_component = -0.3

        quality_component = 0.0
        engaged_count = 0
        aligned_count = 0
        for interceptor in self.interceptorList:
            if interceptor.attacking in (-1, 1):
                continue
            engaged_count += 1
            if interceptor.T_i == threat_index:
                aligned_count += 1
        if engaged_count > 0:
            quality_component = np.clip(2.0 * aligned_count / engaged_count - 1.0, -1.0, 1.0)

        reward = (
            self.reward_weights.height * height_component
            + self.reward_weights.distance * distance_component
            + self.reward_weights.treat * treat_component
            + self.reward_weights.interceptor_usage * interceptor_component
            + self.reward_weights.intercept_quality * quality_component
        )

        return float(np.clip(reward, -4.0, 4.0))
    """
        机动不合规惩罚
    """

    def Punish(self) -> float:
        return -float(self.punish_penalty)

    """
            稀疏奖励
        """

    def SparseReward(self) -> float:
        base = self.reward_weights.sparse
        status = self.escapeFlag

        if status is TerminationStatus.RUNNING:
            dist, _ = self.getClosetMissileDist()
            danger_multiplier = 1.0 if dist <= DANGER_DISTANCE else 0.5
            return -0.05 * base * danger_multiplier * self.getRemainMissileNum()
        if status in (TerminationStatus.FAIL_MISSILE_HIT, TerminationStatus.FAIL_CRASH):
            return -base
        if status in (TerminationStatus.MANEUVER_SUCCESS, TerminationStatus.INTERCEPT_SUCCESS):
            return base
        if status is TerminationStatus.TIMEOUT:
            return -0.2 * base
        return 0.0

    """
            奖励函数：输入当前状态，输出奖励值

            ----------
            Returns

            奖励值Reward
        """

    def rewards(self, state: np.ndarray, actual_goal: int, desired_goal: int) -> float:
        shaped = self.commonReward(state, actual_goal, desired_goal)
        sparse = self.SparseReward()
        total = shaped + sparse
        return float(np.clip(total, -5.0, 5.0))

    '''获取剩余导弹个数'''
    def getRemainMissileNum(self):
        count = 0
        for missile in self.missileList:
            if missile.attacking:
                count += 1
        return count

    '''获取最近的正在来袭的导弹距离及索引'''
    def getClosetMissileDist(self):
        distMin = 10e8
        planePos = [self.aircraftList.X, self.aircraftList.Y, self.aircraftList.Z]
        index = 0
        for i in range(len(self.missileList)):
            if self.missileList[i].attacking:
                targetPos = [self.missileList[i].X, self.missileList[i].Y, self.missileList[i].Z]
                dist = CalDistance(planePos, targetPos)
                if dist < distMin:
                    distMin = dist
                    index = i
        return distMin, index


    """获取动作信息 将动作索引输入 输出三控制量的具体信息和拦截目标"""

    def getActionData(self, action: int) -> Tuple[List[float], int, int]:
        maneuver_idx, desired_goal = self.decode_action(action)
        upper_bound = max(self.missileNum - 1, -1)
        desired_goal = int(np.clip(desired_goal, -1, upper_bound))
        control, actual_goal = self.AirCraftActions(maneuver_idx, desired_goal)
        return control.tolist(), actual_goal, desired_goal

    def step(self, action: int):
        _ = self._get_obs()
        if self.t == 0:
            self.At_1 = action

        control, actual_goal, desired_goal = self.getActionData(action)
        nx, ny, roll, pitch_constraint = control

        launch_allowed = not (self.interceptor_remain == 0 and actual_goal != -1)
        valid_action = (
            self.aircraftList.action_constraint(pitch_constraint)
            and self.aircraftList.speed_constraint(nx)
            and launch_allowed
        )

        speed_flag = True
        info = {}
        if valid_action:
            for _ in range(Gostep):
                state, _, info = self.generate_obs(control)
                if self.escapeFlag is not TerminationStatus.RUNNING:
                    break
            reward = self.rewards(state, actual_goal, desired_goal)
        else:
            for _ in range(Gostep):
                state, _, info = self.constraint_obs(control, speed_flag)
                if self.escapeFlag is not TerminationStatus.RUNNING:
                    break
            reward = self.rewards(state, actual_goal, desired_goal) + self.Punish()
            info = {**info, 'constraint_violation': True}

        if valid_action:
            info = {**info, 'constraint_violation': False}

        if action != self.At_1:
            reward += self.curiosity_scale

        self.t += 1
        self.At_1 = action

        observation = self._genNewState_()
        return observation, reward, self.escapeFlag.value, info


    """对比实验的策略"""

    def compareTest(self, action: int):
        _ = self._get_obs()
        if self.t == 0:
            self.At_1 = action

        forced_goal = 0 if self.t % 2 == 0 else 1
        maneuver_idx, _ = self.decode_action(action)
        control, actual_goal = self.AirCraftActions(maneuver_idx, forced_goal)

        nx, ny, roll, pitch_constraint = control
        launch_allowed = not (self.interceptor_remain == 0 and actual_goal != -1)
        speed_flag = True

        if self.aircraftList.action_constraint(pitch_constraint) and self.aircraftList.speed_constraint(nx) and launch_allowed:
            for _ in range(Gostep):
                state, _, info = self.generate_obs(control)
                if self.escapeFlag is not TerminationStatus.RUNNING:
                    break
            reward = self.rewards(state, actual_goal, forced_goal)
            info = {**info, 'constraint_violation': False}
        else:
            for _ in range(Gostep):
                state, _, info = self.constraint_obs(control, speed_flag)
                if self.escapeFlag is not TerminationStatus.RUNNING:
                    break
            reward = self.rewards(state, actual_goal, forced_goal) + self.Punish()
            info = {**info, 'constraint_violation': True}

        if action != self.At_1:
            reward += self.curiosity_scale

        self.t += 1
        self.At_1 = action

        state = self._genNewState_()
        return state, reward, self.escapeFlag.value, info

    def reset(self):
        missilesNum = self.missileNum
        self.Treat_t = 0
        self.interceptor_remain = self.interceptorNum
        self._set_status(TerminationStatus.RUNNING)
        missileList, aircraftList, planeSpeed, missiles_num, spaceSize, missilesSpeed = reset_para(
            num_missiles=missilesNum)
        self.missileNum = missilesNum
        self.missileSpeed = missilesSpeed
        self.planeSpeed = planeSpeed
        self.missileList = missileList
        self.aircraftList = aircraftList
        self.interceptorList = []
        self.position_scale = 25000.0
        self.target_bucket = self.missileNum + 1
        self.action_space = spaces.Discrete(self.maneuver_action_count * self.target_bucket)
        # 初始化拦截弹列表
        for i in range(self.interceptorNum):
            self.interceptorList.append(Interceptor([self.aircraftList.X, self.aircraftList.Y, self.aircraftList.Z],
                                                    self.aircraftList.V, self.aircraftList.Pitch,
                                                    self.aircraftList.Heading))

        self.observation_planes = np.zeros((1, 6), dtype=np.float32)  # 只有一个飞机

        self.observation_planes[0] = np.array(
            [aircraftList.X, aircraftList.Y, aircraftList.Z, aircraftList.Pitch, aircraftList.Heading,
             aircraftList.roll], dtype=np.float32)

        self.observation_missiles = np.zeros((self.missileNum, 6), dtype=np.float32)

        for i in range(len(missileList)):
            self.observation_missiles[i] = np.array(
                [missileList[i].X, missileList[i].Y, missileList[i].Z, missileList[i].Pitch, missileList[i].Heading, 0],
                dtype=np.float32)
        self.observation_interceptors = np.zeros((self.interceptorNum, 6), dtype=np.float32)

        for i in range(len(self.interceptorList)):
            self.observation_interceptors[i] = np.array(
                [aircraftList.X, aircraftList.Y, aircraftList.Z, aircraftList.Pitch, aircraftList.Heading, 0],
                dtype=np.float32)

        self.StateShape = (
            self.observation_planes.shape[0]
            + self.observation_missiles.shape[0]
            + self.observation_interceptors.shape[0]
        )
        self.D0 = np.empty((missilesNum,), dtype=np.float32)

        for i in range(missilesNum):
            self.D0[i] = CalDistance(
                [self.aircraftList.X, self.aircraftList.Y, self.aircraftList.Z],
                [self.missileList[i].X, self.missileList[i].Y, self.missileList[i].Z],
            )
        self.t = 0
        self.lanchTime = 0
        self.spaceSize = spaceSize
        self.max_steps = max(1, int(spaceSize))

        state = self._genNewState_()
        return state, self.escapeFlag.value, self._build_info()

    # 无量纲化
    '''reverse为TRUE时 量钢化'''
    def normalizeState(self, state, reverse=False):
        if reverse:
            state[:, 0:3] = state[:, 0:3] * self.position_scale
            state[:, 3:6] = state[:, 3:6] * m.pi
        else:
            state[:, 0:3] = state[:, 0:3] / self.position_scale
            state[:, 3:6] = state[:, 3:6] / m.pi
        return state


    """新特征向量的大小"""
    def _getNewStateSpace(self):
        state = np.zeros((self.missileNum * self.interceptorNum + 2 * self.missileNum + self.interceptorNum,),
                         dtype=np.float32)
        return state.shape


    """发射策略"""
    def LanchPolicy(self, interceptor_goal: int) -> bool:
        if interceptor_goal < 0 or interceptor_goal >= self.missileNum:
            return False
        if self.interceptor_remain <= 0:
            return False
        if (self.t - self.lanchTime) < self.launch_gap:
            return False
        if not self.LockConstraint(interceptor_goal):
            return False

        for interceptor in self.interceptorList:
            if interceptor.T_i != -1:
                continue
            interceptor.sync_with_aircraft(
                [self.aircraftList.X, self.aircraftList.Y, self.aircraftList.Z],
                self.aircraftList.Pitch,
                self.aircraftList.Heading,
                self.aircraftList.V,
            )
            launch_speed = max(self.aircraftList.V, self.interceptorSpeed)
            interceptor.begin_pursuit(interceptor_goal, launch_speed)
            self.interceptor_remain -= 1
            self.lanchTime = self.t
            return True

        return False



    """新特征状态"""
    def _genNewState_(self):

        state = np.zeros((self.missileNum * self.interceptorNum + 2 * self.missileNum + self.interceptorNum,), dtype=np.float32)
        missileDist = np.zeros((self.missileNum, ), dtype=np.float32)
        interceptorDist = np.zeros((self.missileNum * self.interceptorNum, ), dtype=np.float32)
        missileStatus = np.zeros((self.missileNum, ), dtype=np.float32)
        interceptorStatus = np.zeros((self.interceptorNum, ), dtype=np.float32)

        for i in range(self.missileNum):
            missileDist[i] = CalDistance([self.aircraftList.X, self.aircraftList.Y, self.aircraftList.Z], [self.missileList[i].X, self.missileList[i].Y, self.missileList[i].Z]) / 10000

        t = 0
        for i in range(self.interceptorNum):
            for j in range(self.missileNum):
                interceptorDist[t] = CalDistance([self.interceptorList[i].X_i, self.interceptorList[i].Y_i, self.interceptorList[i].Z_i], [self.missileList[j].X, self.missileList[j].Y, self.missileList[j].Z]) / 10000
                t += 1

        for i in range(self.missileNum):
            if self.missileList[i].attacking:
                missileStatus[i] = 1
            else:
                missileStatus[i] = -1

        for i in range(self.interceptorNum):
            if self.interceptorList[i].attacking == -1:
                interceptorStatus[i] = -1
            elif self.interceptorList[i].attacking == 1:
                interceptorStatus[i] = 1

            else:
                interceptorStatus[i] = 0

        state = np.concatenate((missileDist, interceptorDist, missileStatus, interceptorStatus))
        return state
