"""可视化脚本：加载指定权重，在固定随机种子下生成三维轨迹GIF动画。"""

import argparse
import os
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch

from DDQN.DDQN import Double_DQN
from DDQN.DQNAgent import MyDQNAgent, device as agent_device
from Environment.init_env import init_env


def set_global_seed(seed: int) -> None:
    """为numpy、random、torch设置随机种子，确保场景可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _create_agent(state_size: int, action_size: int) -> MyDQNAgent:
    """根据状态/动作维度构造一个推理用智能体。"""
    model = Double_DQN(state_size=state_size, action_size=action_size)
    agent = MyDQNAgent(
        model,
        action_size,
        gamma=0.993,
        lr=5e-4,
        e_greed=0.0,
        e_greed_decrement=0.0,
    )
    agent.model.eval()
    agent.target_model.eval()
    return agent


def build_agents(env) -> List[MyDQNAgent]:
    """根据环境类型（单机或多机）构建一个或多个智能体。"""

    if hasattr(env, "get_action_sizes") and callable(getattr(env, "get_action_sizes")):
        action_sizes = [int(size) for size in env.get_action_sizes()]
        state_size = int(env.get_observation_size())
        return [_create_agent(state_size, action_size) for action_size in action_sizes]

    state_size = int(env._getNewStateSpace()[0])
    action_size = int(env._get_actSpace())
    return [_create_agent(state_size, action_size)]


def load_checkpoint(agents: Sequence[MyDQNAgent], checkpoint_path: str) -> None:
    """加载权重，兼容多种保存格式。"""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"未找到权重文件: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=agent_device)

    if isinstance(state, dict):
        # 支持多智能体保存的模型
        keys = [key for key in state.keys() if key.startswith("model_agent_")]
        if keys:
            # 单智能体可直接取首个
            for idx, agent in enumerate(agents):
                key = f"model_agent_{idx}"
                if key not in state:
                    raise KeyError(
                        f"权重文件缺少键 {key}，无法加载第 {idx + 1} 个智能体的参数"
                    )
                try:
                    agent.model.load_state_dict(state[key])
                except RuntimeError as exc:
                    raise RuntimeError(
                        "模型结构与权重不匹配，请确认环境配置与训练时保持一致"
                    ) from exc
                agent.target_model.load_state_dict(agent.model.state_dict())
            return
        elif "model" in state:
            state = state["model"]
        elif "state_dict" in state:
            state = state["state_dict"]

    if len(agents) != 1:
        raise ValueError("权重文件为单智能体格式，但当前环境构造了多个智能体")

    agent = agents[0]
    try:
        agent.model.load_state_dict(state)
    except RuntimeError as exc:
        raise RuntimeError("模型结构与权重不匹配，请确认环境配置与训练时保持一致") from exc
    agent.target_model.load_state_dict(agent.model.state_dict())


def _ensure_array(data: Iterable[float]) -> np.ndarray:
    return np.asarray(list(data), dtype=np.float32)


def _flatten_interceptors(env) -> List:
    if hasattr(env, "interceptor_lists"):
        interceptors = []
        for group in env.interceptor_lists:
            interceptors.extend(group)
        return interceptors
    if hasattr(env, "interceptorList"):
        return list(env.interceptorList)
    return []


@dataclass
class TrajectoryRecorder:
    """收集飞机、导弹与拦截弹的轨迹数据。"""

    aircraft_logs: List[List[np.ndarray]]
    missile_logs: List[List[np.ndarray]]
    interceptor_logs: List[List[np.ndarray]]
    aircraft_labels: List[str]
    missile_labels: List[str]
    interceptor_labels: List[str]

    @classmethod
    def from_env(cls, env) -> "TrajectoryRecorder":
        multi_agent = hasattr(env, "num_agents") and env.num_agents > 1

        if hasattr(env, "aircraftList") and isinstance(env.aircraftList, list):
            aircraft_objects = list(env.aircraftList)
        elif hasattr(env, "aircraftList"):
            aircraft_objects = [env.aircraftList]
        else:
            aircraft_objects = []

        missile_objects = list(getattr(env, "missileList", []))
        interceptor_objects = _flatten_interceptors(env)

        aircraft_logs = [[] for _ in aircraft_objects]
        missile_logs = [[] for _ in missile_objects]
        interceptor_logs = [[] for _ in interceptor_objects]

        if multi_agent:
            aircraft_labels = [f"Aircraft {idx + 1}" for idx in range(len(aircraft_logs))]
            interceptor_labels: List[str] = []
            if hasattr(env, "interceptor_lists"):
                for plane_idx, group in enumerate(env.interceptor_lists):
                    for interceptor_idx, _ in enumerate(group):
                        interceptor_labels.append(
                            f"Interceptor P{plane_idx + 1}-{interceptor_idx + 1}"
                        )
            else:
                interceptor_labels = [
                    f"Interceptor {idx + 1}" for idx in range(len(interceptor_logs))
                ]
        else:
            aircraft_labels = ["Aircraft"] if aircraft_logs else []
            interceptor_labels = [
                f"Interceptor {idx + 1}" for idx in range(len(interceptor_logs))
            ]

        missile_labels = [f"Missile {idx + 1}" for idx in range(len(missile_logs))]

        return cls(
            aircraft_logs=aircraft_logs,
            missile_logs=missile_logs,
            interceptor_logs=interceptor_logs,
            aircraft_labels=aircraft_labels,
            missile_labels=missile_labels,
            interceptor_labels=interceptor_labels,
        )

    def record(self, env) -> None:
        if hasattr(env, "aircraftList") and isinstance(env.aircraftList, list):
            aircraft_objects = env.aircraftList
        elif hasattr(env, "aircraftList"):
            aircraft_objects = [env.aircraftList]
        else:
            aircraft_objects = []

        for idx, plane in enumerate(aircraft_objects):
            self.aircraft_logs[idx].append(
                _ensure_array((plane.X, plane.Y, plane.Z))
            )

        missile_objects = getattr(env, "missileList", [])
        for idx, missile in enumerate(missile_objects):
            self.missile_logs[idx].append(
                _ensure_array((missile.X, missile.Y, missile.Z))
            )

        interceptors = _flatten_interceptors(env)
        for idx, interceptor in enumerate(interceptors):
            self.interceptor_logs[idx].append(
                _ensure_array((interceptor.X_i, interceptor.Y_i, interceptor.Z_i))
            )


def run_episode(
    env,
    agents: Sequence[MyDQNAgent],
    max_steps: Optional[int] = None,
):
    """运行一局对抗，返回轨迹记录、结束标志与提示信息。"""

    multi_agent = hasattr(env, "num_agents") and env.num_agents > 1
    state, done_flag, info = env.reset()
    recorder = TrajectoryRecorder.from_env(env)
    recorder.record(env)

    steps = 0
    while True:
        if multi_agent:
            if not isinstance(state, (list, tuple)):
                state_batch = [state for _ in agents]
            else:
                state_batch = state
            actions = []
            for idx, agent in enumerate(agents):
                current_state = np.asarray(state_batch[idx], dtype=np.float32)
                actions.append(int(agent.predict(current_state)))
            state, _, done_flag, info = env.step(actions)
        else:
            current_state = np.asarray(state if not isinstance(state, (list, tuple)) else state[0], dtype=np.float32)
            action = int(agents[0].predict(current_state))
            state, _, done_flag, info = env.step(action)

        recorder.record(env)
        steps += 1

        if done_flag != -1:
            break
        if max_steps is not None and steps >= max_steps:
            break

    return recorder, done_flag, info


def _line_has_motion(points: Sequence[np.ndarray]) -> bool:
    """判断轨迹是否存在有效运动。"""
    if len(points) <= 1:
        return False
    stacked = np.vstack(points)
    return not np.allclose(stacked[0], stacked, atol=1e-3)


def _collect_all_points(recorder: TrajectoryRecorder) -> np.ndarray:
    """将所有记录的点拼接为数组，用于计算轴范围。"""
    points: List[np.ndarray] = []
    for traj_group in (
        recorder.aircraft_logs,
        recorder.missile_logs,
        recorder.interceptor_logs,
    ):
        for traj in traj_group:
            if traj:
                points.extend(traj)
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.vstack(points)


def _compute_axis_limits(recorder: TrajectoryRecorder):
    """根据所有轨迹点计算坐标轴范围，留出适当边距。"""
    points = _collect_all_points(recorder)
    if points.size == 0:
        return (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)

    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    span = np.maximum(max_vals - min_vals, 1e-3)
    padding = np.maximum(span * 0.05, 1.0)

    xlim = (float(min_vals[0] - padding[0]), float(max_vals[0] + padding[0]))
    ylim = (float(min_vals[1] - padding[1]), float(max_vals[1] + padding[1]))
    zlim = (float(min_vals[2] - padding[2]), float(max_vals[2] + padding[2]))
    return xlim, ylim, zlim


def _setup_axis(ax, title: str, recorder: TrajectoryRecorder) -> None:
    ax.set_title(title)
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y (Altitude) / m")
    ax.set_zlabel("Z / m")

    xlim, ylim, zlim = _compute_axis_limits(recorder)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)


def _update_line(line, points: Sequence[np.ndarray], frame: int):
    if not points:
        return line
    max_index = min(frame + 1, len(points))
    data = np.vstack(points[:max_index])
    line.set_data(data[:, 0], data[:, 1])
    line.set_3d_properties(data[:, 2])
    return line


def save_trajectory_gif(
    recorder: TrajectoryRecorder,
    output_path: Path,
    title: str,
    interval: int = 80,
) -> None:
    """生成三维轨迹随时间演化的GIF动画。"""
    max_frames = 0
    for traj_group in (
        recorder.aircraft_logs,
        recorder.missile_logs,
        recorder.interceptor_logs,
    ):
        for traj in traj_group:
            max_frames = max(max_frames, len(traj))

    if max_frames == 0:
        raise ValueError("无可用于生成动画的轨迹数据")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    plane_colors = ["tab:blue", "tab:orange", "tab:purple", "tab:cyan", "tab:pink", "tab:olive"]

    _setup_axis(ax, title, recorder)

    line_handles = []
    for idx, traj in enumerate(recorder.aircraft_logs):
        color = plane_colors[idx % len(plane_colors)]
        label = (
            recorder.aircraft_labels[idx]
            if idx < len(recorder.aircraft_labels)
            else f"Aircraft {idx + 1}"
        )
        (line,) = ax.plot([], [], [], color=color, linewidth=2.0, label=label)
        line_handles.append((line, traj))

    for idx, traj in enumerate(recorder.missile_logs):
        label = (
            recorder.missile_labels[idx]
            if idx < len(recorder.missile_labels)
            else f"Missile {idx + 1}"
        )
        (line,) = ax.plot([], [], [], linestyle="--", color="tab:red", label=label)
        line_handles.append((line, traj))

    for idx, traj in enumerate(recorder.interceptor_logs):
        label = (
            recorder.interceptor_labels[idx]
            if idx < len(recorder.interceptor_labels)
            else f"Interceptor {idx + 1}"
        )
        (line,) = ax.plot([], [], [], color="tab:green", label=label)
        line_handles.append((line, traj))

    handles, labels = ax.get_legend_handles_labels()
    unique = OrderedDict()
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle
    ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize="small", ncol=2)
    ax.grid(True, alpha=0.3)

    def _animate(frame: int):
        artists = []
        for line, points in line_handles:
            artists.append(_update_line(line, points, frame))
        return artists

    ani = animation.FuncAnimation(
        fig,
        _animate,
        frames=max_frames,
        interval=interval,
        blit=False,
        repeat=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = max(1, int(round(1000.0 / interval)))
    ani.save(str(output_path), writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="加载DDQN权重生成三维轨迹动画")
    parser.add_argument("--checkpoint", required=True, help="DDQN权重文件路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num-missiles", type=int, default=4, help="来袭导弹数量")
    parser.add_argument("--step-num", type=int, default=3500, help="最大仿真步数")
    parser.add_argument("--num-aircraft", type=int, default=2, help="协同飞机数量(<=1表示单机)")
    parser.add_argument(
        "--interceptors",
        type=int,
        default=6,
        help="每架飞机的拦截弹数量（单机模式即总数）",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="可选的最大步数截断，调试用"
    )
    parser.add_argument(
        "--gif", default="outputs/trajectory.gif", help="三维轨迹动画输出路径（.gif）"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    interceptors_total = (
        args.interceptors if args.num_aircraft <= 1 else args.interceptors * args.num_aircraft
    )

    env, _, _ = init_env(
        num_missiles=args.num_missiles,
        StepNum=args.step_num,
        interceptor_num=interceptors_total,
        num_aircraft=args.num_aircraft,
        interceptors_per_plane=args.interceptors if args.num_aircraft > 1 else None,
    )

    agents = build_agents(env)
    load_checkpoint(agents, args.checkpoint)

    recorder, done_flag, info = run_episode(
        env, agents, max_steps=args.max_steps
    )

    ending = {
        -1: "进行中",
        0: "飞机被命中",
        1: "飞机机动逃逸成功",
        2: "拦截成功",
    }.get(done_flag, f"未知状态({done_flag})")
    title = f"Seed={args.seed} | 结果: {ending}"
    if info:
        title = f"{title}\n{info}"

    gif_path = Path(args.gif)
    save_trajectory_gif(recorder, gif_path, title)
    print(f"轨迹动画已保存至: {gif_path}")


if __name__ == "__main__":
    main()
