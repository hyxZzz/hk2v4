"""Evaluate DDQN checkpoints at fixed intervals and export success rates.

This script loads checkpoints produced during training (episodes 100 to 1000
in steps of 100), evaluates each checkpoint for a fixed number of episodes, and
stores the success rates in a CSV file. A deterministic random seed can be
provided to ensure the environment initialisation is reproducible across
checkpoints.
"""

import argparse
import csv
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from utils.validate import (
    EvaluationConfig,
    collect_checkpoints,
    evaluate_checkpoint,
)


TARGET_EPISODES = list(range(100, 1001, 100))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints from episode 100 to 1000",
    )
    parser.add_argument(
        "--model-root",
        default="models",
        help="Directory containing DDQN checkpoints (default: models)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes per checkpoint (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible environment initialisation",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for the CSV results file (default: test_results_seed{seed}.csv)",
    )
    parser.add_argument(
        "--num-missiles",
        type=int,
        default=EvaluationConfig.num_missiles,
        help="Number of incoming missiles used to initialise the environment",
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
        help="Discount factor used by the agent",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=EvaluationConfig.learning_rate,
        help="Learning rate placeholder required by the agent",
    )
    parser.add_argument(
        "--num-aircraft",
        type=int,
        default=EvaluationConfig.num_aircraft,
        help="Number of defending aircraft in the scenario",
    )
    parser.add_argument(
        "--interceptors-per-plane",
        type=int,
        default=EvaluationConfig.interceptors_per_plane,
        help="Number of interceptors carried by each aircraft",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def locate_target_checkpoints(model_root: str) -> Dict[int, str]:
    available: Dict[int, str] = {}
    checkpoints = collect_checkpoints(model_root, start_episode=0)
    for episode, path in checkpoints:
        if episode in TARGET_EPISODES and episode not in available:
            available[episode] = path
    return available


def evaluate_checkpoints(
    checkpoints: Dict[int, str],
    config: EvaluationConfig,
    seed: int,
) -> List[Tuple[int, str, float]]:
    results: List[Tuple[int, str, float]] = []
    for episode in TARGET_EPISODES:
        checkpoint_path = checkpoints.get(episode)
        if not checkpoint_path:
            print(f"Skipping episode {episode}: checkpoint not found")
            continue
        print(f"Evaluating checkpoint from episode {episode} at {checkpoint_path}")
        set_seed(seed)
        success_rate = evaluate_checkpoint(checkpoint_path, config)
        results.append((episode, checkpoint_path, success_rate))
        print(f"Episode {episode}: success rate {success_rate:.4f}")
    return results


def resolve_output_path(seed: int, output: Optional[str]) -> str:
    if output:
        return output
    return f"test_results_seed{seed}.csv"


def save_results(path: str, results: List[Tuple[int, str, float]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["episode", "checkpoint", "success_rate"])
        for episode, checkpoint_path, success_rate in results:
            writer.writerow([episode, checkpoint_path, f"{success_rate:.6f}"])
    print(f"Results saved to {path}")


def main() -> None:
    args = parse_args()
    config = EvaluationConfig(
        episodes=args.episodes,
        num_missiles=args.num_missiles,
        step_num=args.step_num,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        num_aircraft=args.num_aircraft,
        interceptors_per_plane=args.interceptors_per_plane,
    )

    checkpoints = locate_target_checkpoints(args.model_root)
    if not checkpoints:
        print("No matching checkpoints were found.")
        return

    results = evaluate_checkpoints(checkpoints, config, args.seed)
    if not results:
        print("No results to save.")
        return

    output_path = resolve_output_path(args.seed, args.output)
    save_results(output_path, results)


if __name__ == "__main__":
    main()
