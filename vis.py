"""可视化脚本：加载指定权重，在固定随机种子下生成三维轨迹图。"""
import argparse
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
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


def build_agent(state_size: int, action_size: int) -> MyDQNAgent:
    """构造推理用的智能体。"""
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


def load_checkpoint(agent: MyDQNAgent, checkpoint_path: str) -> None:
    """加载权重，兼容多种保存格式。"""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"未找到权重文件: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=agent_device)

    if isinstance(state, dict):
        # 支持多智能体保存的模型
        keys = [key for key in state.keys() if key.startswith("model_agent_")]
        if keys:
            # 单智能体可直接取首个
            agent_key = sorted(keys)[0]
            state = state[agent_key]
        elif "model" in state:
            state = state["model"]
        elif "state_dict" in state:
            state = state["state_dict"]

    agent.model.load_state_dict(state)
    agent.target_model.load_state_dict(agent.model.state_dict())


def record_positions(env, aircraft_log: List[np.ndarray], missile_logs: List[List[np.ndarray]],
                     interceptor_logs: List[List[np.ndarray]]) -> None:
    """记录当前时刻飞机、导弹与拦截弹的位置。"""
    aircraft_log.append(env.observation_planes[0, :3].copy())

    for idx in range(env.missileNum):
        missile_logs[idx].append(env.observation_missiles[idx, :3].copy())

    for idx in range(len(env.interceptorList)):
        interceptor_logs[idx].append(env.observation_interceptors[idx, :3].copy())


def run_episode(env, agent: MyDQNAgent, max_steps: Optional[int] = None) -> Tuple[
    List[np.ndarray], List[List[np.ndarray]], List[List[np.ndarray]], int, str
]:
    """运行一局对抗，返回轨迹与结束状态。"""
    state, _, _ = env.reset()

    aircraft_log: List[np.ndarray] = []
    missile_logs: List[List[np.ndarray]] = [[] for _ in range(env.missileNum)]
    interceptor_logs: List[List[np.ndarray]] = [[] for _ in range(len(env.interceptorList))]

    record_positions(env, aircraft_log, missile_logs, interceptor_logs)

    done_flag = -1
    info = ""
    steps = 0

    while True:
        if isinstance(state, (list, tuple)):
            # 单智能体环境仍可能返回列表，取第一个即可
            current_state = state[0]
        else:
            current_state = state

        current_state = np.asarray(current_state, dtype=np.float32)
        action = int(agent.predict(current_state))
        state, _, done_flag, info = env.step(action)
        record_positions(env, aircraft_log, missile_logs, interceptor_logs)

        steps += 1
        if done_flag != -1:
            break
        if max_steps is not None and steps >= max_steps:
            break

    return aircraft_log, missile_logs, interceptor_logs, done_flag, info


def _line_has_motion(points: Sequence[np.ndarray]) -> bool:
    """判断轨迹是否存在有效运动。"""
    if len(points) <= 1:
        return False
    stacked = np.vstack(points)
    return not np.allclose(stacked[0], stacked, atol=1e-3)


def plot_trajectories(
    aircraft_log: Sequence[np.ndarray],
    missile_logs: Sequence[Sequence[np.ndarray]],
    interceptor_logs: Sequence[Sequence[np.ndarray]],
    output_path: Path,
    title: str,
) -> None:
    """绘制三维轨迹并保存。"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    aircraft_points = np.vstack(aircraft_log)
    ax.plot(
        aircraft_points[:, 0],
        aircraft_points[:, 1],
        aircraft_points[:, 2],
        color="tab:blue",
        label="Aircraft",
        linewidth=2.0,
    )
    ax.scatter(
        aircraft_points[0, 0],
        aircraft_points[0, 1],
        aircraft_points[0, 2],
        color="tab:blue",
        marker="o",
        s=60,
        label="Aircraft Start",
    )
    ax.scatter(
        aircraft_points[-1, 0],
        aircraft_points[-1, 1],
        aircraft_points[-1, 2],
        color="tab:blue",
        marker="^",
        s=60,
        label="Aircraft End",
    )

    for idx, traj in enumerate(missile_logs):
        if not _line_has_motion(traj):
            continue
        missile_points = np.vstack(traj)
        ax.plot(
            missile_points[:, 0],
            missile_points[:, 1],
            missile_points[:, 2],
            linestyle="--",
            label=f"Missile {idx + 1}",
        )

    for idx, traj in enumerate(interceptor_logs):
        if not _line_has_motion(traj):
            continue
        interceptor_points = np.vstack(traj)
        ax.plot(
            interceptor_points[:, 0],
            interceptor_points[:, 1],
            interceptor_points[:, 2],
            color="tab:green",
            label=f"Interceptor {idx + 1}",
        )

    ax.set_title(title)
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y (Altitude) / m")
    ax.set_zlabel("Z / m")

    handles, labels = ax.get_legend_handles_labels()
    unique = OrderedDict()
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle
    ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize="small", ncol=2)
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="加载DDQN权重生成三维轨迹图")
    parser.add_argument("--checkpoint", required=True, help="DDQN权重文件路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num-missiles", type=int, default=4, help="来袭导弹数量")
    parser.add_argument("--step-num", type=int, default=3500, help="最大仿真步数")
    parser.add_argument("--interceptors", type=int, default=6, help="拦截弹数量")
    parser.add_argument(
        "--output", default="outputs/trajectory.png", help="三维轨迹图输出路径"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="可选的最大步数截断，调试用"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    env, _, _ = init_env(
        num_missiles=args.num_missiles,
        StepNum=args.step_num,
        interceptor_num=args.interceptors,
        num_aircraft=1,
    )

    state_size = env._getNewStateSpace()[0]
    action_size = env._get_actSpace()
    agent = build_agent(state_size, action_size)
    load_checkpoint(agent, args.checkpoint)

    aircraft_log, missile_logs, interceptor_logs, done_flag, info = run_episode(
        env, agent, max_steps=args.max_steps
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

    output_path = Path(args.output)
    plot_trajectories(aircraft_log, missile_logs, interceptor_logs, output_path, title)
    print(f"轨迹图已保存至: {output_path}")


if __name__ == "__main__":
    main()
