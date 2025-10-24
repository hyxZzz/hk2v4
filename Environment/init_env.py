from Environment.env import ManeuverEnv, CooperativeManeuverEnv
import numpy as np
import math as m
from utils.common import ComputeHeading, ComputePitch
from flat_models.trajectory import Aircraft, Missiles
from Environment.reset_env import reset_cooperative_para


# 环境的来袭导弹数num_missiles，最大步数StepNum

def init_env(
    num_missiles=4,
    StepNum=3500,
    interceptor_num=6,
    num_aircraft=2,
    interceptors_per_plane=None,
):
    if num_aircraft <= 1:
        # 飞机的初始位置，x和y为[-10000, 10000], z为服从2000为均值，300为标准差的正态分布

        # 飞行高度8~12Km
        a_x = np.random.uniform(-10000, 10000)
        a_y = np.random.uniform(-10000, 10000)
        a_z = np.random.uniform(8000, 12000)

        # 飞行速度0.5~1.2马赫
        a_v = np.random.uniform(0.5, 1.2)
        a_v = a_v * 340

        # 飞机的俯仰角和偏转角
        aPitch = 0
        aHeading = np.random.uniform(-1, 1)
        aHeading = aHeading * m.pi

        # 在水平椭圆区域内生成导弹位置
        angles = np.random.uniform(0, 2 * m.pi, size=num_missiles)
        radial_scale = np.random.uniform(0.85, 1.15, size=num_missiles)
        major_axis = 20000.0
        minor_axis = 15000.0
        missile_x = a_x + major_axis * radial_scale * np.cos(angles)
        missile_y = a_y + minor_axis * radial_scale * np.sin(angles)
        altitude_offsets = np.random.uniform(-3000.0, 3000.0, size=num_missiles)
        missile_z = np.clip(a_z + altitude_offsets, 0.0, None)
        mposList = []
        mHeadingList = []
        mPitchList = []
        for i in range(len(missile_x)):
            mposList.append([missile_x[i], missile_z[i], missile_y[i]])
            mHeadingList.append(ComputeHeading([a_x, a_z, a_y], [missile_x[i], missile_z[i], missile_y[i]]))
            mPitchList.append(ComputePitch([a_x, a_z, a_y], [missile_x[i], missile_z[i], missile_y[i]]))

        # 导弹速度不低于2马赫
        m_v = np.random.uniform(2, 3)
        m_v = m_v * 340

        aircraft_agent = [Aircraft([a_x, a_z, a_y], V=a_v, Pitch=aPitch, Heading=aHeading)]
        missiles_list = []
        for i in range(len(mposList)):
            missiles_list.append(Missiles(mposList[i], V=m_v, Pitch=mPitchList[i], Heading=mHeadingList[i]))

        env = ManeuverEnv(
            missiles_list,
            aircraft_agent[0],
            planeSpeed=a_v,
            missilesNum=num_missiles,
            spaceSize=StepNum,
            InterceptorNum=interceptor_num,
        )

        return env, aircraft_agent, missiles_list

    interceptors_per_plane = interceptors_per_plane or interceptor_num
    missiles_list, aircraft_list, _, missile_speed, _ = reset_cooperative_para(
        num_aircraft=num_aircraft,
        num_missiles=num_missiles,
        interceptors_per_plane=interceptors_per_plane,
        StepNum=StepNum,
    )

    env = CooperativeManeuverEnv(
        missiles_list,
        aircraft_list,
        missilesNum=num_missiles,
        spaceSize=StepNum,
        missilesSpeed=missile_speed,
        interceptors_per_plane=interceptors_per_plane,
    )

    return env, aircraft_list, missiles_list
