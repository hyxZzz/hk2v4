# coding: utf-8
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import math as m
import numpy as np
from gym import spaces

from Environment.ActionDepository import getActionDepository
from Environment.reset_env import reset_para
from flat_models.ThreatEvaluate import CalTreat
from flat_models.trajectory import (
    Aircraft,
    Interceptor,
    Missiles,
    MaxInterceptorDist,
    MinInterceptorDist,
)
from utils.common import CalDistance

act_num = 29
Gostep = 1
INTERCEPT_SUCCESS_DISTANCE = 20.0
MISSILE_HIT_DISTANCE = 20.0
PLANE_FEATURE_DIM = 8
MISSILE_FEATURE_DIM = 8
INTERCEPTOR_FEATURE_DIM = 8
STATUS_READY = 0
STATUS_IN_FLIGHT = 1
STATUS_HIT = 2
STATUS_LOST = 3
PLANE_SAFE_ALTITUDE = (8000.0, 12000.0)
POSITION_SCALE = 25000.0
ANGLE_SCALE = m.pi
VELOCITY_SCALE = 1000.0
TEAM_REWARD_WEIGHTS = dict(intercept=3.0, remaining=-2.0)
SUCCESS_COMPLETION_BONUS = 5.0
TIMEOUT_PENALTY = 4.0
LOSS_PENALTY = 6.0
ACTIVE_MISSILE_STEP_PENALTY = 0.5
ATTRIBUTION_BONUS = 0.1
DANGER_DISTANCE = 3000.0
DANGER_HOLD_PENALTY = 0.8
RESPONSIVE_LAUNCH_BONUS = 0.3
PREMATURE_LAUNCH_PENALTY = 0.5
INVALID_TARGET_PENALTY = 0.6
CONSTRAINT_FAILURE_PENALTY = 0.6
MAX_PREMATURE_DISTANCE = DANGER_DISTANCE * 3.0
WITHIN_RANGE_LAUNCH_BONUS = 0.35
OUT_OF_RANGE_LAUNCH_PENALTY = 0.7
MIN_INTERCEPT_DISTANCE = float(MinInterceptorDist)
MAX_INTERCEPT_DISTANCE = float(MaxInterceptorDist)
LanchGap = 70


def _encode_status(status: str) -> int:
    if status == "ready":
        return STATUS_READY
    if status == "in_flight":
        return STATUS_IN_FLIGHT
    if status == "hit":
        return STATUS_HIT
    return STATUS_LOST


class ManeuverEnv:
    def __init__(
        self,
        missileList: Sequence[Missiles],
        aircraftList: Sequence[Aircraft],
        planeSpeed: Sequence[float],
        missilesNum: int = 4,
        spaceSize: int = 3500,
        missilesSpeed: float = 680.0,
        InterceptorNum: int = 12,
        interceptorSpeed: float = 540.0,
        allow_cross_lock: bool = True,
        max_locks_per_missile: int = 2,
    ):
        self.action_dep = getActionDepository(missilesNum, act_num)
        self.action_space = spaces.Discrete(len(self.action_dep))
        self.allow_cross_lock = allow_cross_lock
        self.max_locks_per_missile = max_locks_per_missile

        self.missileNum = missilesNum
        self.spaceSize = spaceSize
        self.missileSpeed = missilesSpeed
        self.interceptorNum = InterceptorNum
        self.interceptorSpeed = interceptorSpeed
        self.planeSpeed = list(planeSpeed)
        self.num_planes = len(aircraftList)
        self.interceptor_remain = InterceptorNum

        self.missileList: List[Missiles] = list(missileList)
        self.aircraftList: List[Aircraft] = list(aircraftList)
        self.interceptorList: List[Interceptor] = []
        self.interceptor_pools: Dict[int, List[int]] = {}
        self.borrowed_usage_count = 0
        self.intercepts_by_owner = np.zeros(self.num_planes, dtype=np.int32)

        self.observation_planes = np.zeros((self.num_planes, PLANE_FEATURE_DIM), dtype=np.float32)
        self.observation_missiles = np.zeros((self.missileNum, MISSILE_FEATURE_DIM), dtype=np.float32)
        self.observation_interceptors = np.zeros((self.interceptorNum, INTERCEPTOR_FEATURE_DIM), dtype=np.float32)

        self.t = 0
        self.escapeFlag = -1
        self.lanchTime = np.full(self.num_planes, -LanchGap, dtype=np.int32)
        self.team_reward_log: List[float] = []
        self.attribution_bonus = np.zeros(self.num_planes, dtype=np.float32)
        self._step_metadata: List[Dict[str, object]] = [self._empty_step_meta() for _ in range(self.num_planes)]

        self._init_interceptors()
        self._update_observations()

    # ------------------------------------------------------------------
    # Environment lifecycle
    # ------------------------------------------------------------------
    def reset(self):
        missiles, aircraft, plane_speeds, missiles_num, space_size, missile_speed = reset_para(
            num_missiles=self.missileNum,
            StepNum=self.spaceSize,
            num_planes=self.num_planes,
            interceptor_num=self.interceptorNum,
        )

        self.missileNum = missiles_num
        self.spaceSize = space_size
        self.missileSpeed = missile_speed
        self.planeSpeed = plane_speeds
        self.missileList = missiles
        self.aircraftList = aircraft

        self.interceptorList.clear()
        self.interceptor_pools.clear()
        self.borrowed_usage_count = 0
        self.intercepts_by_owner = np.zeros(self.num_planes, dtype=np.int32)
        self.team_reward_log.clear()
        self.attribution_bonus = np.zeros(self.num_planes, dtype=np.float32)
        self.t = 0
        self.escapeFlag = -1
        self.lanchTime = np.full(self.num_planes, -LanchGap, dtype=np.int32)
        self.interceptor_remain = self.interceptorNum

        self.observation_planes = np.zeros((self.num_planes, PLANE_FEATURE_DIM), dtype=np.float32)
        self.observation_missiles = np.zeros((self.missileNum, MISSILE_FEATURE_DIM), dtype=np.float32)
        self.observation_interceptors = np.zeros((self.interceptorNum, INTERCEPTOR_FEATURE_DIM), dtype=np.float32)

        self.action_dep = getActionDepository(self.missileNum, act_num)
        self.action_space = spaces.Discrete(len(self.action_dep))
        self._step_metadata = [self._empty_step_meta() for _ in range(self.num_planes)]

        self._init_interceptors()
        self._update_observations()
        state = self._build_global_state()
        return state, self.escapeFlag, "Go on Combating..."

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _init_interceptors(self):
        per_plane = self.interceptorNum // max(1, self.num_planes)
        remainder = self.interceptorNum % max(1, self.num_planes)
        counter = 0
        for plane_id in range(self.num_planes):
            self.interceptor_pools[plane_id] = []
            share = per_plane + (1 if plane_id < remainder else 0)
            for _ in range(share):
                aircraft = self.aircraftList[plane_id]
                interceptor = Interceptor(
                    [aircraft.X, aircraft.Y, aircraft.Z],
                    aircraft.V,
                    aircraft.Pitch,
                    aircraft.Heading,
                    owner=plane_id,
                )
                self.interceptorList.append(interceptor)
                self.interceptor_pools[plane_id].append(counter)
                counter += 1
        while counter < self.interceptorNum:
            aircraft = self.aircraftList[0]
            interceptor = Interceptor(
                [aircraft.X, aircraft.Y, aircraft.Z],
                aircraft.V,
                aircraft.Pitch,
                aircraft.Heading,
                owner=0,
            )
            self.interceptorList.append(interceptor)
            self.interceptor_pools.setdefault(0, []).append(counter)
            counter += 1
        self.interceptor_remain = self.interceptorNum

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------
    def _update_observations(self):
        self.observation_planes.fill(0.0)
        self.observation_missiles.fill(0.0)
        self.observation_interceptors.fill(0.0)

        for plane_id, aircraft in enumerate(self.aircraftList):
            self.observation_planes[plane_id] = np.array(
                [
                    aircraft.X,
                    aircraft.Y,
                    aircraft.Z,
                    aircraft.V,
                    aircraft.Pitch,
                    aircraft.Heading,
                    aircraft.roll,
                    1.0,
                ],
                dtype=np.float32,
            )

        for missile_id, missile in enumerate(self.missileList):
            active_flag = 1.0 if missile.attacking else 0.0
            self.observation_missiles[missile_id] = np.array(
                [
                    missile.X,
                    missile.Y,
                    missile.Z,
                    missile.V,
                    missile.Pitch,
                    missile.Heading,
                    active_flag,
                    0.0,
                ],
                dtype=np.float32,
            )

        for idx, interceptor in enumerate(self.interceptorList):
            if idx >= self.interceptorNum:
                break
            status_code = float(_encode_status(interceptor.status))
            owner = float(interceptor.owner)
            self.observation_interceptors[idx] = np.array(
                [
                    interceptor.X_i,
                    interceptor.Y_i,
                    interceptor.Z_i,
                    interceptor.V_i,
                    interceptor.Pitch_i,
                    interceptor.Heading_i,
                    status_code,
                    owner,
                ],
                dtype=np.float32,
            )

    def _build_global_state(self) -> np.ndarray:
        state_components = [
            self.observation_planes.reshape(-1),
            self.observation_missiles.reshape(-1),
            self.observation_interceptors.reshape(-1),
        ]
        state = np.concatenate(state_components).astype(np.float32)
        return self.normalizeState(state)

    # ------------------------------------------------------------------
    # Core environment updates
    # ------------------------------------------------------------------
    def step(self, actions: Sequence[int]):
        if len(actions) != self.num_planes:
            raise ValueError(f"Expect {self.num_planes} actions, received {len(actions)}")

        self.attribution_bonus = np.zeros(self.num_planes, dtype=np.float32)
        info = {"attribution_bonus": self.attribution_bonus}
        self.escapeFlag = -1
        self._step_metadata = [self._empty_step_meta() for _ in range(self.num_planes)]

        for _ in range(Gostep):
            self._apply_actions(actions)
            self._update_missiles()
            self._update_interceptors()
            self._check_collisions()
            self.t += 1
            if self.escapeFlag != -1:
                break

        self._update_observations()
        state = self._build_global_state()
        reward = self._compute_reward()
        info = {
            "attribution_bonus": self.attribution_bonus.copy(),
            "intercepts_by_owner": self.intercepts_by_owner.copy(),
            "borrowed_usage_count": self.borrowed_usage_count,
            "action_metadata": [dict(meta) for meta in self._step_metadata],
        }

        done_flag = self.escapeFlag
        if done_flag == -1 and self.t >= self.spaceSize:
            self.escapeFlag = 1
            done_flag = 1

        return state, reward, done_flag, info

    def _apply_actions(self, actions: Sequence[int]):
        for plane_id, action_idx in enumerate(actions):
            action_idx = int(action_idx)
            meta = self._step_metadata[plane_id]
            meta["action_index"] = action_idx
            meta["launch"] = False
            meta["constraint_failed"] = False
            meta["invalid_target"] = False
            meta["target_distance"] = None
            meta["nearest_distance"] = self._nearest_active_missile_distance(plane_id)
            meta["danger_zone"] = (
                meta["nearest_distance"] is not None and meta["nearest_distance"] < DANGER_DISTANCE
            )
            meta["range_invalid"] = False

            action = self.action_dep[action_idx]
            nx, ny, roll, pitch_constraint = action[:4]
            target_cmd = int(round(action[4])) if action.shape[0] > 4 else -1
            meta["target"] = target_cmd

            aircraft = self.aircraftList[plane_id]

            launch = False
            valid_target = 0 <= target_cmd < len(self.missileList) and self.missileList[target_cmd].attacking
            if valid_target:
                meta["target_distance"] = CalDistance(
                    [aircraft.X, aircraft.Y, aircraft.Z],
                    [
                        self.missileList[target_cmd].X,
                        self.missileList[target_cmd].Y,
                        self.missileList[target_cmd].Z,
                    ],
                )
                if not aircraft.LimitCondition(
                    [
                        self.missileList[target_cmd].X,
                        self.missileList[target_cmd].Y,
                        self.missileList[target_cmd].Z,
                    ]
                ):
                    meta["range_invalid"] = True
                launch = self._attempt_launch(plane_id, target_cmd, meta)
                if not launch and not meta.get("range_invalid", False):
                    self._prepare_lock_only(plane_id, target_cmd)
            else:
                if target_cmd >= 0:
                    meta["invalid_target"] = True
                if (
                    0 <= target_cmd < len(self.missileList)
                    and self.missileList[target_cmd].attacking
                ):
                    self._prepare_lock_only(plane_id, target_cmd)

            if not aircraft.action_constraint(pitch_constraint) or not aircraft.speed_constraint(nx):
                nx = 0
                ny = 1
                roll = aircraft.roll

            aircraft.AircraftPostition(None, nx, ny, roll, pitch_constraint)

    def _nearest_active_missile_distance(self, plane_id: int) -> Optional[float]:
        plane = self.aircraftList[plane_id]
        distances = [
            CalDistance([plane.X, plane.Y, plane.Z], [missile.X, missile.Y, missile.Z])
            for missile in self.missileList
            if missile.attacking
        ]
        if not distances:
            return None
        return float(min(distances))

    def _attempt_launch(self, plane_id: int, target_id: int, meta: Optional[Dict[str, object]] = None) -> bool:
        if target_id < 0:
            return False
        if self.t - self.lanchTime[plane_id] < LanchGap:
            if meta is not None:
                meta["constraint_failed"] = True
            return False
        if not self.LockConstraint(target_id):
            if meta is not None:
                meta["constraint_failed"] = True
            return False
        missile = self.missileList[target_id]
        plane = self.aircraftList[plane_id]
        target_pos = [missile.X, missile.Y, missile.Z]
        if not plane.LimitCondition(target_pos):
            if meta is not None:
                meta["constraint_failed"] = True
                meta["range_invalid"] = True
                if meta.get("target_distance") is None:
                    meta["target_distance"] = CalDistance(
                        [plane.X, plane.Y, plane.Z], target_pos
                    )
            return False
        allocated = self._allocate_interceptor(plane_id)
        if allocated is None:
            if meta is not None:
                meta["constraint_failed"] = True
            return False
        interceptor = self.interceptorList[allocated]
        interceptor.sync_with_aircraft([plane.X, plane.Y, plane.Z], plane.Pitch, plane.Heading, plane.V)
        launch_speed = max(plane.V, self.interceptorSpeed)
        interceptor.begin_pursuit(target_id, launch_speed, plane_id, float(self.t))
        self.lanchTime[plane_id] = self.t
        if meta is not None:
            meta["launch"] = True
            meta["constraint_failed"] = False
        return True

    def _prepare_lock_only(self, plane_id: int, target_id: int):
        for idx in self.interceptor_pools.get(plane_id, []):
            interceptor = self.interceptorList[idx]
            if interceptor.attacking == -1:
                interceptor.T_i = target_id

    def _allocate_interceptor(self, plane_id: int) -> Optional[int]:
        ready_pool = self.interceptor_pools.get(plane_id, [])
        ready_pool = [idx for idx in ready_pool if self.interceptorList[idx].attacking == -1]
        if ready_pool:
            interceptor_index = ready_pool.pop(0)
            self.interceptor_pools[plane_id] = ready_pool
            self.interceptor_remain = max(0, self.interceptor_remain - 1)
            return interceptor_index
        if not self.allow_cross_lock:
            return None
        for other_id, pool in self.interceptor_pools.items():
            if other_id == plane_id:
                continue
            candidates = [idx for idx in pool if self.interceptorList[idx].attacking == -1]
            for idx in candidates:
                interceptor = self.interceptorList[idx]
                if plane_id in interceptor.available_to:
                    pool.remove(idx)
                    self.borrowed_usage_count += 1
                    interceptor.current_holder = plane_id
                    self.interceptor_remain = max(0, self.interceptor_remain - 1)
                    return idx
        for other_id, pool in self.interceptor_pools.items():
            if other_id == plane_id:
                continue
            candidates = [idx for idx in pool if self.interceptorList[idx].attacking == -1]
            if candidates:
                idx = candidates[0]
                pool.remove(idx)
                interceptor = self.interceptorList[idx]
                interceptor.set_available_to([interceptor.owner, plane_id])
                interceptor.current_holder = plane_id
                self.borrowed_usage_count += 1
                self.interceptor_remain = max(0, self.interceptor_remain - 1)
                return idx
        return None

    def _update_missiles(self):
        for missile in self.missileList:
            if not missile.attacking:
                continue
            target_plane = self._select_plane_for_missile(missile)
            plane = self.aircraftList[target_plane]
            missile.MissilePosition([plane.X, plane.Y, plane.Z], plane.V, plane.Pitch, plane.Heading)

    def _select_plane_for_missile(self, missile: Missiles) -> int:
        closest_plane = 0
        closest_dist = float("inf")
        for plane_id, plane in enumerate(self.aircraftList):
            dist = CalDistance([plane.X, plane.Y, plane.Z], [missile.X, missile.Y, missile.Z])
            if dist < closest_dist:
                closest_dist = dist
                closest_plane = plane_id
        return closest_plane

    def _update_interceptors(self):
        for idx, interceptor in enumerate(self.interceptorList):
            if interceptor.attacking == -1:
                holder = interceptor.current_holder
                plane = self.aircraftList[holder]
                interceptor.sync_with_aircraft([plane.X, plane.Y, plane.Z], plane.Pitch, plane.Heading, plane.V)
                continue
            if interceptor.attacking == 1:
                continue
            target_idx = interceptor.T_i
            if target_idx < 0 or target_idx >= len(self.missileList):
                interceptor.mark_failure()
                continue
            missile = self.missileList[target_idx]
            position = interceptor.InterceptorPosition([missile.X, missile.Y, missile.Z], missile.V, missile.Pitch, missile.Heading)
            dist = CalDistance(position, [missile.X, missile.Y, missile.Z])
            if dist < INTERCEPT_SUCCESS_DISTANCE:
                missile.attacking = False
                interceptor.mark_hit(float(self.t))
                owner = interceptor.owner
                if 0 <= owner < len(self.intercepts_by_owner):
                    self.intercepts_by_owner[owner] += 1
                    self.attribution_bonus[owner] += ATTRIBUTION_BONUS

    def _check_collisions(self):
        for missile in self.missileList:
            if not missile.attacking:
                continue
            for plane_id, plane in enumerate(self.aircraftList):
                dist = CalDistance([plane.X, plane.Y, plane.Z], [missile.X, missile.Y, missile.Z])
                if dist < MISSILE_HIT_DISTANCE:
                    self.escapeFlag = 0
                    return
        if all(not missile.attacking for missile in self.missileList):
            self.escapeFlag = 2

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------
    def _compute_reward(self) -> float:
        active_missiles = sum(1 for missile in self.missileList if missile.attacking)
        remaining_ratio = active_missiles / max(1, self.missileNum)
        intercept_ratio = 1.0 - remaining_ratio
        threat = 0.0
        for missile in self.missileList:
            if not missile.attacking:
                continue
            plane_id = self._select_plane_for_missile(missile)
            plane = self.aircraftList[plane_id]
            plane_state = [plane.X, plane.Y, plane.Z, plane.Pitch, plane.Heading, plane.roll]
            missile_state = [missile.X, missile.Y, missile.Z, missile.Pitch, missile.Heading, 0.0]
            threat = max(threat, CalTreat(plane_state, missile_state, plane.V, missile.V))
        threat_penalty = min(threat / 100.0, 1.0)

        reward = TEAM_REWARD_WEIGHTS["intercept"] * intercept_ratio
        reward += TEAM_REWARD_WEIGHTS["remaining"] * remaining_ratio
        reward -= ACTIVE_MISSILE_STEP_PENALTY * remaining_ratio

        if self.escapeFlag == 2:
            reward += SUCCESS_COMPLETION_BONUS
        elif self.escapeFlag == 1:
            reward -= TIMEOUT_PENALTY
        elif self.escapeFlag == 0:
            reward -= LOSS_PENALTY

        reward -= 0.3 * threat_penalty
        reward += float(np.sum(self.attribution_bonus))
        reward += self._launch_shaping_reward()
        self.team_reward_log.append(reward)
        return reward

    def _launch_shaping_reward(self) -> float:
        shaping_reward = 0.0
        penalties = 0.0
        for plane_id, meta in enumerate(self._step_metadata):
            nearest = self._nearest_active_missile_distance(plane_id)
            if nearest is None:
                nearest = meta.get("nearest_distance")
            else:
                meta["nearest_distance"] = nearest

            target_idx = meta.get("target", -1)
            target_distance = None
            if 0 <= target_idx < len(self.missileList):
                missile = self.missileList[target_idx]
                if missile.attacking:
                    plane = self.aircraftList[plane_id]
                    target_distance = CalDistance(
                        [plane.X, plane.Y, plane.Z],
                        [missile.X, missile.Y, missile.Z],
                    )
            if target_distance is None:
                target_distance = meta.get("target_distance")
            else:
                meta["target_distance"] = target_distance

            danger_zone = nearest is not None and nearest < DANGER_DISTANCE
            meta["danger_zone"] = danger_zone
            launch = bool(meta.get("launch"))
            range_invalid = bool(meta.get("range_invalid"))
            invalid_target = bool(meta.get("invalid_target"))
            constraint_failed = bool(meta.get("constraint_failed"))

            if danger_zone:
                if launch and target_distance is not None:
                    ratio = 1.0 - min(target_distance, DANGER_DISTANCE) / DANGER_DISTANCE
                    shaping_reward += RESPONSIVE_LAUNCH_BONUS * max(ratio, 0.0)
                elif not launch and nearest is not None:
                    ratio = 1.0 - min(nearest, DANGER_DISTANCE) / DANGER_DISTANCE
                    penalties += DANGER_HOLD_PENALTY * max(ratio, 0.0)
            else:
                if launch:
                    far_distance = target_distance if target_distance is not None else nearest
                    if far_distance is None:
                        far_distance = MAX_PREMATURE_DISTANCE
                    excess = max(0.0, far_distance - DANGER_DISTANCE)
                    penalties += PREMATURE_LAUNCH_PENALTY * min(
                        excess / max(1.0, MAX_PREMATURE_DISTANCE - DANGER_DISTANCE),
                        1.0,
                    )

            if launch and not range_invalid and target_distance is not None:
                within_span = MAX_INTERCEPT_DISTANCE - MIN_INTERCEPT_DISTANCE
                if within_span > 0:
                    clipped = min(max(target_distance - MIN_INTERCEPT_DISTANCE, 0.0), within_span)
                    ratio = 1.0 - clipped / within_span
                else:
                    ratio = 1.0
                shaping_reward += WITHIN_RANGE_LAUNCH_BONUS * max(ratio, 0.0)
            if launch and range_invalid:
                penalties += OUT_OF_RANGE_LAUNCH_PENALTY

            if invalid_target:
                penalties += INVALID_TARGET_PENALTY
            if constraint_failed:
                penalties += CONSTRAINT_FAILURE_PENALTY

        return shaping_reward - penalties

    def _mean_missile_distance(self) -> float:
        distances = []
        for missile in self.missileList:
            if not missile.attacking:
                continue
            plane_id = self._select_plane_for_missile(missile)
            plane = self.aircraftList[plane_id]
            distances.append(CalDistance([plane.X, plane.Y, plane.Z], [missile.X, missile.Y, missile.Z]))
        if not distances:
            return DANGER_DISTANCE
        return float(np.mean(distances))

    # ------------------------------------------------------------------
    # Constraints and utilities
    # ------------------------------------------------------------------
    def LockConstraint(self, missile_id: int) -> bool:
        if missile_id < 0:
            return False
        locked = 0
        for interceptor in self.interceptorList:
            if interceptor.T_i == missile_id and interceptor.status in ("ready", "in_flight"):
                locked += 1
        return locked < self.max_locks_per_missile

    def normalizeState(self, state: np.ndarray) -> np.ndarray:
        state = state.copy()
        total_plane = self.num_planes * PLANE_FEATURE_DIM
        total_missile = self.missileNum * MISSILE_FEATURE_DIM
        plane_slice = state[:total_plane].reshape(self.num_planes, PLANE_FEATURE_DIM)
        missile_slice = state[total_plane:total_plane + total_missile].reshape(self.missileNum, MISSILE_FEATURE_DIM)
        interceptor_slice = state[total_plane + total_missile:].reshape(self.interceptorNum, INTERCEPTOR_FEATURE_DIM)

        plane_slice[:, 0:3] /= POSITION_SCALE
        plane_slice[:, 3] /= VELOCITY_SCALE
        plane_slice[:, 4:7] /= ANGLE_SCALE

        missile_slice[:, 0:3] /= POSITION_SCALE
        missile_slice[:, 3] /= VELOCITY_SCALE
        missile_slice[:, 4:6] /= ANGLE_SCALE

        interceptor_slice[:, 0:3] /= POSITION_SCALE
        interceptor_slice[:, 3] /= VELOCITY_SCALE
        interceptor_slice[:, 4:6] /= ANGLE_SCALE
        interceptor_slice[:, 6] /= STATUS_LOST
        interceptor_slice[:, 7] /= max(1, self.num_planes - 1)

        return np.concatenate(
            [
                plane_slice.reshape(-1),
                missile_slice.reshape(-1),
                interceptor_slice.reshape(-1),
            ]
        )

    def get_local_obs(self, plane_id: int) -> np.ndarray:
        plane = self.observation_planes[plane_id]
        missiles = []
        for missile in self.missileList:
            relative = np.array([
                missile.X - plane[0],
                missile.Y - plane[1],
                missile.Z - plane[2],
                missile.V,
                missile.Pitch,
                missile.Heading,
                1.0 if missile.attacking else 0.0,
            ], dtype=np.float32)
            missiles.append(relative)
        missiles_arr = np.array(missiles, dtype=np.float32)
        interceptors = []
        for idx in self.interceptor_pools.get(plane_id, []):
            interceptor = self.interceptorList[idx]
            interceptors.append(np.array([
                interceptor.X_i - plane[0],
                interceptor.Y_i - plane[1],
                interceptor.Z_i - plane[2],
                interceptor.V_i,
                interceptor.Pitch_i,
                interceptor.Heading_i,
                float(_encode_status(interceptor.status)),
                float(interceptor.owner),
            ], dtype=np.float32))
        if interceptors:
            interceptors_arr = np.array(interceptors, dtype=np.float32)
        else:
            interceptors_arr = np.zeros((0, INTERCEPTOR_FEATURE_DIM), dtype=np.float32)
        return np.concatenate(
            [
                plane,
                missiles_arr.reshape(-1),
                interceptors_arr.reshape(-1),
            ]
        )

    # Legacy API compatibility ------------------------------------------------
    def _get_obs(self):
        return np.concatenate(
            [
                self.observation_planes,
                self.observation_missiles,
                self.observation_interceptors,
            ]
        )

    def _get_actSpace(self):
        return len(self.action_dep)

    def _getNewStateSpace(self):
        state = self._build_global_state()
        return state.shape

    def getRemainMissileNum(self):
        return sum(1 for missile in self.missileList if missile.attacking)

    def render(self):
        pass

    def _empty_step_meta(self) -> Dict[str, object]:
        return {
            "action_index": -1,
            "target": -1,
            "launch": False,
            "constraint_failed": False,
            "invalid_target": False,
            "target_distance": None,
            "nearest_distance": None,
            "danger_zone": False,
            "range_invalid": False,
        }

