"""Validation script for evaluating DDQN checkpoints with multi-agent heads."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Sequence, Tuple

import torch
from tensorboardX import SummaryWriter

from DDQN.DDQN import Double_DQN
from DDQN.DQNAgent import MyDQNAgent, device as agent_device
from Environment.init_env import init_env


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration values used during validation."""

    episodes: int = 100
    num_missiles: int = 4
    num_planes: int = 2
    step_num: int = 3500
    gamma: float = 0.993
    learning_rate: float = 5e-4


EPISODE_PATTERN = re.compile(r"DDQN_episode(\d+)\.pth$")


def collect_checkpoints(model_root: str, start_episode: int) -> List[Tuple[int, str]]:
    """Return a sorted list of checkpoint paths and their episode numbers."""

    checkpoints: List[Tuple[int, str]] = []
    if not os.path.isdir(model_root):
        return checkpoints

    for root, _, files in os.walk(model_root):
        for filename in files:
            match = EPISODE_PATTERN.search(filename)
            if not match:
                continue
            episode = int(match.group(1))
            if episode < start_episode:
                continue
            checkpoints.append((episode, os.path.join(root, filename)))

    checkpoints.sort(key=lambda item: item[0])
    return checkpoints


def build_agent(state_size: int, action_size: int, num_agents: int, config: EvaluationConfig) -> MyDQNAgent:
    """Construct a ``MyDQNAgent`` ready for evaluation."""

    model = Double_DQN(state_size=state_size, action_size_each=action_size, num_agents=num_agents)
    agent = MyDQNAgent(
        model,
        action_size,
        num_agents=num_agents,
        gamma=config.gamma,
        lr=config.learning_rate,
        e_greed=0.0,
        e_greed_decrement=0.0,
    )
    agent.model.eval()
    agent.target_model.eval()
    return agent


def load_checkpoint(agent: MyDQNAgent, checkpoint_path: str) -> None:
    """Load model parameters from ``checkpoint_path`` into ``agent``."""

    state = torch.load(checkpoint_path, map_location=agent_device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    agent.model.load_state_dict(state)
    agent.target_model.load_state_dict(agent.model.state_dict())


def select_action(agent: MyDQNAgent, state):
    """Return the greedy multi-agent action for ``state`` using ``agent``'s policy."""

    return agent.predict(state)


def evaluate_checkpoint(
    checkpoint_path: str,
    config: EvaluationConfig,
) -> float:
    """Evaluate ``checkpoint_path`` and return the intercept success rate."""

    env, _, _ = init_env(
        num_missiles=config.num_missiles,
        StepNum=config.step_num,
        num_planes=config.num_planes,
        interceptor_num=12,
    )
    action_size = env._get_actSpace()
    state_size = env._getNewStateSpace()[0]
    agent = build_agent(state_size, action_size, env.num_planes, config)
    load_checkpoint(agent, checkpoint_path)

    success_count = 0
    for _ in range(config.episodes):
        state, done_flag, _ = env.reset()
        while True:
            action = select_action(agent, state)
            state, _, done_flag, _ = env.step(action)
            if done_flag != -1:
                if done_flag == 2:
                    success_count += 1
                break

    return success_count / float(config.episodes)


def create_writer() -> Tuple[SummaryWriter, str]:
    """Create a ``SummaryWriter`` under ``runs/val`` and return it and its path."""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("runs", "val", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    return writer, log_dir


def save_csv(log_dir: str, results: Sequence[Tuple[int, str, float]]) -> str:
    """Persist evaluation results as a CSV file and return its path."""

    csv_path = os.path.join(log_dir, "intercept_success_rates.csv")
    with open(csv_path, "w", encoding="utf-8") as csv_file:
        csv_file.write("episode,checkpoint,success_rate\n")
        for episode, checkpoint, success_rate in results:
            csv_file.write(f"{episode},{checkpoint},{success_rate:.6f}\n")
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate DDQN checkpoints")
    parser.add_argument(
        "--model-root",
        default="models",
        help="Root directory that stores DDQN checkpoints (default: models)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=EvaluationConfig.episodes,
        help="Number of evaluation episodes per checkpoint (default: 100)",
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=100,
        help="Minimum checkpoint episode index to evaluate (default: 100)",
    )
    parser.add_argument(
        "--num-missiles",
        type=int,
        default=EvaluationConfig.num_missiles,
        help="Number of incoming missiles used to initialise the environment",
    )
    parser.add_argument(
        "--num-planes",
        type=int,
        default=EvaluationConfig.num_planes,
        help="Number of defending aircraft in the environment",
    )
    parser.add_argument(
        "--step-num",
        type=int,
        default=EvaluationConfig.step_num,
        help="Maximum number of steps per episode for the environment",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=EvaluationConfig.gamma,
        help="Discount factor used by the agent (default: 0.99)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=EvaluationConfig.learning_rate,
        help="Learning rate placeholder required by the agent (default: 0.001)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = EvaluationConfig(
        episodes=args.episodes,
        num_missiles=args.num_missiles,
        num_planes=args.num_planes,
        step_num=args.step_num,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
    )

    writer, log_dir = create_writer()
    checkpoints = collect_checkpoints(args.model_root, args.start_episode)
    results: List[Tuple[int, str, float]] = []

    try:
        for episode, checkpoint_path in checkpoints:
            success_rate = evaluate_checkpoint(checkpoint_path, config)
            writer.add_scalar("intercept_success_rate", success_rate, episode)
            results.append((episode, checkpoint_path, success_rate))
            print(f"Evaluated checkpoint {checkpoint_path}: success rate {success_rate:.4f}")
    finally:
        writer.close()

    csv_path = save_csv(log_dir, results)
    print(f"Validation results saved to {csv_path}")


if __name__ == "__main__":
    main()
