#!/usr/bin/env python3
"""可视化三维轨迹并导出 GIF 动画的脚本。

脚本复用训练与验证阶段所采用的 ``ManeuverEnv`` 环境配置，
加载训练过程中生成的权重文件，依据策略执行一次完整仿真，
记录飞机、来袭导弹与拦截弹的三维轨迹，并使用 ``matplotlib``
生成 GIF 动画后保存到指定路径。
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # 使用无窗口后端，便于服务器或批处理环境运行

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch

from DDQN.DQNAgent import MyDQNAgent
from Environment.init_env import init_env
from utils.validate import (
    EvaluationConfig,
    build_agent as build_evaluation_agent,
    load_checkpoint as load_evaluation_checkpoint,
    select_action as select_policy_action,
)

# 环境完成标志与说明文本映射
DONE_MESSAGES = {
    -1: "未完成：达到最大步数或仍在运行",
    0: "失败：防守飞机被击中",
    1: "失败：达到场景最大时长",
    2: "成功：全部来袭导弹被拦截",
}


@dataclass
class FrameData:
    """用于存储单帧轨迹数据的结构。"""

    planes: np.ndarray
    missiles: np.ndarray
    missile_active: np.ndarray
    interceptors: np.ndarray
    interceptor_status: List[str]


def parse_args() -> argparse.Namespace:
    """解析命令行参数，并忽略训练脚本专有的多余选项。"""

    parser = argparse.ArgumentParser(description="生成空战场景的三维轨迹 GIF")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="训练阶段生成的模型权重 (.pth) 文件路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("trajectory.gif"),
        help="GIF 输出路径（默认：trajectory.gif）",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3500,
        help="单轮仿真的最大步数（默认：3500）",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="动画播放帧率（默认：20）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子，便于结果复现（默认：随机）",
    )
    parser.add_argument(
        "--num-missiles",
        type=int,
        default=4,
        help="初始化场景的来袭导弹数量（默认：4）",
    )
    parser.add_argument(
        "--num-planes",
        type=int,
        default=2,
        help="防守飞机数量（默认：2）",
    )
    parser.add_argument(
        "--num-interceptors",
        type=int,
        default=12,
        help="可用拦截弹数量（默认：12）",
    )
    parser.add_argument(
        "--step-num",
        type=int,
        default=3500,
        help="环境的最大步数 StepNum（默认：3500，与训练配置保持一致）",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.993,
        help="训练所使用的折扣因子 γ（默认：0.993）",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="训练时采用的学习率（默认：5e-4）",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(
            "警告：检测到以下未使用的参数 {}，已忽略。该脚本仅需加载权重并执行可视化。".format(
                " ".join(unknown)
            )
        )
    return args


def capture_frame(env) -> FrameData:
    """从环境对象提取当前帧的空间信息。"""

    plane_positions = np.array([[ac.X, ac.Y, ac.Z] for ac in env.aircraftList], dtype=float)
    missile_positions = np.array([[ms.X, ms.Y, ms.Z] for ms in env.missileList], dtype=float)
    missile_active = np.array([bool(ms.attacking) for ms in env.missileList], dtype=bool)
    interceptor_positions = np.array([[ic.X_i, ic.Y_i, ic.Z_i] for ic in env.interceptorList], dtype=float)
    interceptor_status = [ic.status for ic in env.interceptorList]
    return FrameData(
        planes=plane_positions,
        missiles=missile_positions,
        missile_active=missile_active,
        interceptors=interceptor_positions,
        interceptor_status=interceptor_status,
    )


@dataclass
class EpisodeSummary:
    steps: int
    total_reward: float


def build_agent(env, gamma: float, learning_rate: float) -> MyDQNAgent:
    """根据训练与验证阶段相同的网络结构构建智能体。"""

    action_size = env._get_actSpace()
    state_size = env._getNewStateSpace()[0]
    config = EvaluationConfig(
        episodes=1,
        num_missiles=env.missilesNum,
        num_planes=env.num_planes,
        step_num=env.spaceSize,
        gamma=gamma,
        learning_rate=learning_rate,
    )
    agent = build_evaluation_agent(state_size, action_size, env.num_planes, config)
    return agent


def simulate_episode(
    env,
    agent: MyDQNAgent,
    max_steps: int,
) -> Tuple[List[FrameData], int, EpisodeSummary]:
    """执行一次策略仿真，返回轨迹、完成标志和简单统计。"""

    state, done_flag, _ = env.reset()
    frames: List[FrameData] = [capture_frame(env)]
    total_reward = 0.0
    step_count = 0

    for _ in range(max_steps):
        actions = select_policy_action(agent, state)
        state, reward, done_flag, _ = env.step(actions)
        frames.append(capture_frame(env))
        total_reward += float(reward)
        step_count += 1
        if done_flag != -1:
            break

    summary = EpisodeSummary(steps=step_count, total_reward=total_reward)
    return frames, done_flag, summary


def stack_frame_data(frames: Sequence[FrameData]):
    """将帧列表转换为便于绘图的张量。"""

    plane_tracks = np.stack([frame.planes for frame in frames], axis=0)
    missile_tracks = np.stack([frame.missiles for frame in frames], axis=0)
    missile_active = np.stack([frame.missile_active for frame in frames], axis=0)
    interceptor_tracks = np.stack([frame.interceptors for frame in frames], axis=0)
    interceptor_status = np.empty((len(frames), interceptor_tracks.shape[1]), dtype=object)
    for idx, frame in enumerate(frames):
        interceptor_status[idx, :] = frame.interceptor_status
    return plane_tracks, missile_tracks, missile_active, interceptor_tracks, interceptor_status


def compute_axis_limits(arrays: Sequence[np.ndarray]) -> Tuple[float, float, float, float, float, float]:
    """根据多组三维坐标计算统一的坐标轴范围。"""

    coords = [arr.reshape(-1, 3) for arr in arrays if arr.size > 0]
    if not coords:
        return -1.0, 1.0, -1.0, 1.0, -1.0, 1.0
    concat = np.concatenate(coords, axis=0)
    concat = concat[~np.isnan(concat).any(axis=1)]
    if concat.size == 0:
        return -1.0, 1.0, -1.0, 1.0, -1.0, 1.0
    min_vals = concat.min(axis=0)
    max_vals = concat.max(axis=0)
    center = (min_vals + max_vals) / 2.0
    span = (max_vals - min_vals).max() / 2.0
    span = max(span, 1.0)
    x_lim = (center[0] - span, center[0] + span)
    y_lim = (center[1] - span, center[1] + span)
    z_lim = (center[2] - span, center[2] + span)
    return x_lim[0], x_lim[1], y_lim[0], y_lim[1], z_lim[0], z_lim[1]


def create_animation(
    frames: Sequence[FrameData],
    done_flag: int,
    output_path: Path,
    fps: int,
) -> None:
    """使用给定的轨迹数据生成并保存 GIF 动画。"""

    (
        plane_tracks,
        missile_tracks,
        missile_active,
        interceptor_tracks,
        interceptor_status,
    ) = stack_frame_data(frames)

    num_frames = plane_tracks.shape[0]
    num_planes = plane_tracks.shape[1]
    num_missiles = missile_tracks.shape[1]
    num_interceptors = interceptor_tracks.shape[1]

    (
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
    ) = compute_axis_limits([plane_tracks, missile_tracks, interceptor_tracks])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.view_init(elev=25, azim=-60)

    plane_lines = []
    plane_markers = []
    for idx in range(num_planes):
        label = "防守飞机" if idx == 0 else "_nolegend_"
        line, = ax.plot([], [], [], color="#1f77b4", lw=2.0, label=label)
        marker, = ax.plot([], [], [], marker="o", color="#1f77b4", markersize=6)
        plane_lines.append(line)
        plane_markers.append(marker)

    missile_lines = []
    missile_markers = []
    for idx in range(num_missiles):
        label = "来袭导弹" if idx == 0 else "_nolegend_"
        line, = ax.plot([], [], [], color="#d62728", lw=1.5, alpha=0.85, label=label)
        marker, = ax.plot([], [], [], marker="^", color="#d62728", markersize=5)
        missile_lines.append(line)
        missile_markers.append(marker)

    interceptor_lines = []
    interceptor_markers = []
    for idx in range(num_interceptors):
        label = "拦截弹" if idx == 0 else "_nolegend_"
        line, = ax.plot([], [], [], color="#2ca02c", lw=1.2, alpha=0.6, label=label)
        marker, = ax.plot([], [], [], marker="s", color="#2ca02c", markersize=4)
        interceptor_lines.append(line)
        interceptor_markers.append(marker)

    ax.legend(loc="upper right")
    title = ax.set_title("三维空战轨迹")
    status_text = ax.text2D(0.02, 0.90, "", transform=ax.transAxes)

    interceptor_color_map = {
        "ready": ("#2ca02c", 0.2),
        "in_flight": ("#2ca02c", 1.0),
        "hit": ("#9467bd", 0.8),
        "lost": ("#7f7f7f", 0.6),
    }

    def update(frame_idx: int):
        title.set_text(f"三维空战轨迹 - Step {frame_idx}")
        running = frame_idx < num_frames - 1 or done_flag == -1
        if running:
            status_text.set_text("状态：进行中")
        else:
            status_text.set_text(f"状态：{DONE_MESSAGES.get(done_flag, '已结束')}")

        for idx, line in enumerate(plane_lines):
            history = plane_tracks[: frame_idx + 1, idx]
            line.set_data(history[:, 0], history[:, 1])
            line.set_3d_properties(history[:, 2])
            marker = plane_markers[idx]
            marker.set_data([plane_tracks[frame_idx, idx, 0]], [plane_tracks[frame_idx, idx, 1]])
            marker.set_3d_properties([plane_tracks[frame_idx, idx, 2]])

        for idx, line in enumerate(missile_lines):
            history = missile_tracks[: frame_idx + 1, idx]
            line.set_data(history[:, 0], history[:, 1])
            line.set_3d_properties(history[:, 2])
            marker = missile_markers[idx]
            marker.set_data([missile_tracks[frame_idx, idx, 0]], [missile_tracks[frame_idx, idx, 1]])
            marker.set_3d_properties([missile_tracks[frame_idx, idx, 2]])
            active = bool(missile_active[frame_idx, idx])
            alpha = 0.95 if active else 0.25
            line.set_alpha(alpha)
            marker.set_alpha(alpha)

        for idx, line in enumerate(interceptor_lines):
            history = interceptor_tracks[: frame_idx + 1, idx]
            line.set_data(history[:, 0], history[:, 1])
            line.set_3d_properties(history[:, 2])
            marker = interceptor_markers[idx]
            marker.set_data([interceptor_tracks[frame_idx, idx, 0]], [interceptor_tracks[frame_idx, idx, 1]])
            marker.set_3d_properties([interceptor_tracks[frame_idx, idx, 2]])
            status = interceptor_status[frame_idx, idx]
            color, alpha = interceptor_color_map.get(status, ("#2ca02c", 0.4))
            line.set_color(color)
            marker.set_color(color)
            line.set_alpha(alpha)
            marker.set_alpha(alpha)

        return (
            plane_lines
            + plane_markers
            + missile_lines
            + missile_markers
            + interceptor_lines
            + interceptor_markers
            + [title, status_text]
        )

    anim = FuncAnimation(fig, update, frames=num_frames, interval=1000 / max(fps, 1), blit=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    env, _, _ = init_env(
        num_missiles=args.num_missiles,
        StepNum=max(args.step_num, 1),
        interceptor_num=args.num_interceptors,
        num_planes=args.num_planes,
    )

    agent = build_agent(env, gamma=args.gamma, learning_rate=args.learning_rate)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"未找到模型权重文件：{args.checkpoint}")
    load_evaluation_checkpoint(agent, str(args.checkpoint))

    max_steps = max(1, min(args.max_steps, env.spaceSize))

    frames, done_flag, summary = simulate_episode(env, agent, max_steps)
    create_animation(frames, done_flag, args.output, fps=max(args.fps, 1))

    done_message = DONE_MESSAGES.get(done_flag, "仿真已结束")
    print(
        "仿真完成：共执行 {} 步，累计奖励 {:.2f}，终止状态：{}。".format(
            summary.steps, summary.total_reward, done_message
        )
    )
    print(f"GIF 已保存至：{args.output.resolve()}")


if __name__ == "__main__":
    main()
