#!/usr/bin/env python3
"""可视化脚本

该脚本加载指定的模型权重，运行一次预测环境，
并根据得到的状态序列绘制飞机、导弹和拦截弹的三维轨迹。
"""
from __future__ import annotations

import argparse
import os
from typing import Iterable, List, Tuple

import matplotlib

# 使用非交互式后端以便在无显示环境下运行
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.predict import predictResult


ENTITY_PLANE_LABEL = "飞机"
ENTITY_MISSILE_LABEL = "来袭导弹"
ENTITY_INTERCEPTOR_LABEL = "拦截弹"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="加载DDQN权重并生成三维轨迹图")
    parser.add_argument(
        "model_path",
        type=str,
        help="模型权重文件路径（.pth或.pt）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="trajectory.png",
        help="保存图片的路径，默认为trajectory.png",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="输出图片的分辨率（DPI）",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="生成图像后在窗口中展示（可能需要图形界面环境）",
    )
    return parser


def _escape_flag_to_text(flag: int) -> str:
    mapping = {
        -1: "任务尚未结束",
        0: "任务失败：被击中或坠毁",
        1: "任务成功：机动逃逸",
        2: "任务成功：拦截完成",
    }
    return mapping.get(flag, f"未知状态({flag})")


def _collect_trajectories(
    states: np.ndarray,
    missile_num: int,
    interceptor_num: int,
) -> List[Tuple[str, np.ndarray]]:
    """按照实体类别拆分并返回轨迹数据。"""
    trajectories: List[Tuple[str, np.ndarray]] = []
    if states.ndim != 3 or states.shape[2] < 3:
        raise ValueError("状态序列形状异常，预期为[时间, 实体数, >=6]")

    plane_traj = states[:, 0, :3]
    trajectories.append((ENTITY_PLANE_LABEL, plane_traj))

    for idx in range(missile_num):
        label = f"{ENTITY_MISSILE_LABEL}{idx + 1}"
        trajectories.append((label, states[:, 1 + idx, :3]))

    base_idx = 1 + missile_num
    for idx in range(interceptor_num):
        label = f"{ENTITY_INTERCEPTOR_LABEL}{idx + 1}"
        trajectories.append((label, states[:, base_idx + idx, :3]))

    return trajectories


def _plot_trajectories(
    trajectories: Iterable[Tuple[str, np.ndarray]],
    title: str,
    dpi: int,
    output_path: str,
    show: bool,
) -> None:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.cm.get_cmap("tab20")
    for idx, (label, coords) in enumerate(trajectories):
        coords = np.asarray(coords)
        if coords.size == 0:
            continue
        # 过滤全零的时间步，避免绘制无效轨迹
        mask = ~np.all(np.isclose(coords, 0.0), axis=1)
        if not np.any(mask):
            continue
        coords = coords[mask]
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], label=label, color=colors(idx % colors.N))
        ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2], color=colors(idx % colors.N), marker="o", s=20)
        ax.scatter(coords[-1, 0], coords[-1, 1], coords[-1, 2], color=colors(idx % colors.N), marker="x", s=40)

    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_zlabel("Z / m")
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    model_path = os.path.abspath(args.model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"找不到模型权重文件：{model_path}")

    # 运行一次预测获取轨迹数据
    (steps,
     state_sequence,
     _treat_values,
     env_snapshot,
     _state_copy,
     remain_interceptors,
     escape_flag) = predictResult(model_path)

    valid_length = min(steps + 1, state_sequence.shape[0])
    states = state_sequence[:valid_length]

    trajectories = _collect_trajectories(
        states,
        missile_num=getattr(env_snapshot, "missileNum", 0),
        interceptor_num=getattr(env_snapshot, "interceptorNum", 0),
    )

    title = (
        f"三维轨迹 - {_escape_flag_to_text(escape_flag)}\n"
        f"剩余拦截弹：{remain_interceptors}"
    )

    _plot_trajectories(
        trajectories=trajectories,
        title=title,
        dpi=args.dpi,
        output_path=os.path.abspath(args.output) if args.output else "",
        show=args.show,
    )


if __name__ == "__main__":
    main()
