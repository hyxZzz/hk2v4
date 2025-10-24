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
    step_num: int = 3500
    gamma: float = 0.993
    learning_rate: float = 5e-4
    num_aircraft: int = 2
    interceptors_per_plane: int = 6


EPISODE_PATTERN = re.compile(r"DDQN_episode(\d+)\.pth$|DDQN_multi_episode(\d+)\.pth$")


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
            episode_candidates = [int(group) for group in match.groups() if group]
            if not episode_candidates:
                continue
            episode = episode_candidates[0]
            if episode < start_episode:
                continue
            checkpoints.append((episode, os.path.join(root, filename)))

    checkpoints.sort(key=lambda item: item[0])
    return checkpoints


def build_agents(state_size: int, action_sizes: Sequence[int], config: EvaluationConfig) -> List[MyDQNAgent]:
    agents: List[MyDQNAgent] = []
    for action_size in action_sizes:
        model = Double_DQN(state_size=state_size, action_size=action_size)
        agent = MyDQNAgent(
            model,
            action_size,
            gamma=config.gamma,
            lr=config.learning_rate,
            e_greed=0.0,
            e_greed_decrement=0.0,
        )
        agent.model.eval()
        agent.target_model.eval()
        agents.append(agent)
    return agents


def load_checkpoint(agents: Sequence[MyDQNAgent], checkpoint_path: str) -> None:
    state = torch.load(checkpoint_path, map_location=agent_device)

    if isinstance(state, dict):
        loaded = False
        for idx, agent in enumerate(agents):
            key = f"model_agent_{idx}"
            if key in state:
                agent.model.load_state_dict(state[key])
                agent.target_model.load_state_dict(agent.model.state_dict())
                loaded = True
        if loaded:
            return
        if "model" in state and len(agents) == 1:
            state = state["model"]

    for agent in agents:
        agent.model.load_state_dict(state)
        agent.target_model.load_state_dict(agent.model.state_dict())


def select_action(agent: MyDQNAgent, state) -> int:
    state_tensor = torch.tensor(state, dtype=torch.float32, device=agent_device)
    with torch.no_grad():
        q_values = agent.model(state_tensor)
    return int(q_values.argmax())


def evaluate_checkpoint(
    checkpoint_path: str,
    config: EvaluationConfig,
) -> float:
    env, _, _ = init_env(
        num_missiles=config.num_missiles,
        StepNum=config.step_num,
        num_aircraft=config.num_aircraft,
        interceptors_per_plane=config.interceptors_per_plane,
    )

    if hasattr(env, 'get_action_sizes'):
        action_sizes = env.get_action_sizes()
        state_size = env.get_observation_size()
        agents = build_agents(state_size, action_sizes, config)

        load_checkpoint(agents, checkpoint_path)
        success_count = 0
        for _ in range(config.episodes):
            states, done_flag, _ = env.reset()
            if isinstance(states, tuple):
                states = states[0]
            # Cooperative environment returns list of states per agent
            if isinstance(states, list):
                agent_states = states
            else:
                agent_states = [states for _ in agents]

            while True:
                actions = [select_action(agent, agent_states[idx]) for idx, agent in enumerate(agents)]
                next_states, _, done_flag, _ = env.step(actions)
                if isinstance(next_states, list):
                    agent_states = next_states
                else:
                    agent_states = [next_states for _ in agents]
                if done_flag != -1:
                    if done_flag == 2:
                        success_count += 1
                    break
        return success_count / float(config.episodes)

    # 兼容旧版单智能体环境
    action_size = env._get_actSpace()
    state_size = env._getNewStateSpace()[0]
    agent = build_agents(state_size, [action_size], config)[0]
    load_checkpoint([agent], checkpoint_path)

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
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("runs", "val", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    return writer, log_dir


def save_csv(log_dir: str, results: Sequence[Tuple[int, str, float]]) -> str:
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

    checkpoints = collect_checkpoints(args.model_root, args.start_episode)
    if not checkpoints:
        print("No checkpoints found under", args.model_root)
        return

    writer, log_dir = create_writer()
    results = []
    for episode, path in checkpoints:
        success_rate = evaluate_checkpoint(path, config)
        writer.add_scalar('intercept_success_rate', success_rate, episode)
        results.append((episode, path, success_rate))
        print(f"Episode {episode}: success rate {success_rate:.4f}")

    csv_path = save_csv(log_dir, results)
    print('Results saved to', csv_path)


if __name__ == "__main__":
    main()
