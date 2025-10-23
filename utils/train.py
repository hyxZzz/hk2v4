import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
from tensorboardX import SummaryWriter

from DDQN.DDQN import Double_DQN
from DDQN.DQNAgent import MyDQNAgent
from Environment.init_env import init_env
from utils.ERbuffer import MyMemoryBuffer
from utils.validate import (
    EvaluationConfig,
    create_writer as create_validation_writer,
    evaluate_checkpoint,
    save_csv as save_validation_csv,
)

writer = SummaryWriter('./models/DQNmodels/DDQNmodels3_23/runs/train_process_multi_agent')


@dataclass
class EvaluationMetrics:
    mean_total_reward: float
    mean_reward_per_step: float
    success_rate: float


def run_train_episode(
    agents: Sequence[MyDQNAgent],
    env,
    replay_buffers: Sequence[MyMemoryBuffer],
    shared_buffers: Dict[int, MyMemoryBuffer],
    agent_action_sizes: Sequence[int],
    memory_warmup_size: int,
    learn_freq: int,
    batch_size: int,
):
    num_agents = len(agents)
    total_rewards = np.zeros(num_agents, dtype=np.float32)
    train_losses = [0.0 for _ in range(num_agents)]
    unique_action_sizes = tuple(sorted(set(agent_action_sizes)))

    states, _, _ = env.reset()
    step = 0

    while True:
        step += 1
        actions = [agent.sample(states[i]) for i, agent in enumerate(agents)]
        next_states, rewards, done_flag, _ = env.step(actions)

        for idx in range(num_agents):
            experience = (states[idx], actions[idx], rewards[idx], next_states[idx], done_flag)
            replay_buffers[idx].add(experience)
            shared_buffers[agent_action_sizes[idx]].add(experience)

        can_learn = step % learn_freq == 0 and all(
            shared_buffers[action_size].size() > memory_warmup_size for action_size in unique_action_sizes
        )

        if can_learn:
            for idx, agent in enumerate(agents):
                shared_buffer = shared_buffers[agent_action_sizes[idx]]
                if shared_buffer.size() <= memory_warmup_size:
                    continue
                experiences = shared_buffer.sample(batch_size)
                if not experiences:
                    continue
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*experiences)
                loss = agent.learn(
                    batch_state, batch_action, batch_reward, batch_next_state, batch_done
                )
                train_losses[idx] = float(loss.detach().cpu().item() if torch.is_tensor(loss) else loss)

        total_rewards += np.array(rewards, dtype=np.float32)
        states = next_states

        if done_flag != -1:
            break

    return total_rewards, train_losses


def evaluate_agents(
    agents: Sequence[MyDQNAgent],
    env,
    eval_episodes: int = 10,
) -> EvaluationMetrics:
    total_rewards: List[float] = []
    per_step_rewards: List[float] = []
    intercept_successes = 0

    previous_states = []
    for agent in agents:
        previous_states.append(
            (
                agent.model.training,
                agent.target_model.training,
                agent.e_greed,
            )
        )
        agent.model.eval()
        agent.target_model.eval()
        agent.e_greed = 0.0

    try:
        for _ in range(eval_episodes):
            states, _, _ = env.reset()
            episode_reward = 0.0
            steps = 0
            success = False

            while True:
                actions = [agent.predict(states[i]) for i, agent in enumerate(agents)]
                states, rewards, done_flag, _ = env.step(actions)
                steps += 1
                episode_reward += float(np.mean(rewards))

                if done_flag != -1:
                    if done_flag == 2:
                        success = True
                    break

            total_rewards.append(episode_reward)
            per_step_rewards.append(episode_reward / max(steps, 1))
            if success:
                intercept_successes += 1
    finally:
        for agent, (model_mode, target_mode, epsilon) in zip(agents, previous_states):
            agent.model.train(model_mode)
            agent.target_model.train(target_mode)
            agent.e_greed = epsilon

    mean_total_reward = float(np.mean(total_rewards)) if total_rewards else 0.0
    mean_reward_per_step = float(np.mean(per_step_rewards)) if per_step_rewards else 0.0
    success_rate = intercept_successes / float(eval_episodes) if eval_episodes > 0 else 0.0

    return EvaluationMetrics(
        mean_total_reward=mean_total_reward,
        mean_reward_per_step=mean_reward_per_step,
        success_rate=success_rate,
    )


def main():
    parser = argparse.ArgumentParser(description='multi-agent cooperative defence training')

    parser.add_argument('--memory_size', type=int, default=60000, help='Size of replay memory')
    parser.add_argument('--memory_warmup_size', type=int, default=4000, help='Warmup size of replay memory')
    parser.add_argument('--learn_freq', type=int, default=1, help='Frequency of learning updates')
    parser.add_argument('--batch_size', type=int, default=384, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for training')
    parser.add_argument('--gamma', type=float, default=0.993, help='Discount factor')
    parser.add_argument(
        '--target_update_freq',
        type=int,
        default=15,
        help='Number of learning steps between target network updates',
    )
    parser.add_argument('--max_episode', type=int, default=1000, help='Maximum number of episodes')
    parser.add_argument(
        '--validation_episodes',
        type=int,
        default=EvaluationConfig.episodes,
        help='Number of validation episodes for each checkpoint',
    )

    args = parser.parse_args()

    num_missiles = 4
    step_num = 3500
    num_aircraft = 2
    interceptors_per_plane = 6

    env, _, _ = init_env(
        num_missiles=num_missiles,
        StepNum=step_num,
        num_aircraft=num_aircraft,
        interceptors_per_plane=interceptors_per_plane,
    )

    action_sizes = [int(size) for size in env.get_action_sizes()]
    state_size = env.get_observation_size()

    replay_buffers = [MyMemoryBuffer(args.memory_size) for _ in range(num_aircraft)]
    shared_buffers = {
        action_size: MyMemoryBuffer(args.memory_size) for action_size in set(action_sizes)
    }

    epsilon_start = 0.85
    epsilon_end = 0.05
    decay_ratio = 0.2
    decay_target_episodes = max(1, int(args.max_episode * decay_ratio))
    decay_steps = max(1, int(step_num * decay_target_episodes))
    epsilon_decrement = (epsilon_start - epsilon_end) / decay_steps

    agents: List[MyDQNAgent] = []
    for action_size in action_sizes:
        model = Double_DQN(state_size=state_size, action_size=action_size)
        agent = MyDQNAgent(
            model,
            action_size,
            gamma=args.gamma,
            lr=args.learning_rate,
            e_greed=epsilon_start,
            e_greed_decrement=epsilon_decrement,
            min_epsilon=epsilon_end,
            update_target_steps=args.target_update_freq,
        )
        agents.append(agent)

    validation_config = EvaluationConfig(
        episodes=args.validation_episodes,
        num_missiles=num_missiles,
        step_num=step_num,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        num_aircraft=num_aircraft,
        interceptors_per_plane=interceptors_per_plane,
    )
    val_writer, val_log_dir = create_validation_writer()
    validation_results = []
    validation_csv_path = None

    start_time = time.time()
    print('start training...')
    episode = 0
    max_episode = args.max_episode

    while episode < max_episode:
        for _ in range(50):
            total_reward, train_losses = run_train_episode(
                agents,
                env,
                replay_buffers,
                shared_buffers,
                action_sizes,
                args.memory_warmup_size,
                args.learn_freq,
                args.batch_size,
            )
            mean_reward = float(np.mean(total_reward))
            mean_loss = float(np.mean(train_losses))
            writer.add_scalar('train/mean_reward', mean_reward, episode)
            writer.add_scalar('train/mean_loss', mean_loss, episode)
            episode += 1
            if episode >= max_episode:
                break

        if episode % 50 == 0:
            eval_metrics = evaluate_agents(agents, env, eval_episodes=args.validation_episodes)
            writer.add_scalar('eval/mean_total_reward', eval_metrics.mean_total_reward, episode)
            writer.add_scalar('eval/mean_reward_per_step', eval_metrics.mean_reward_per_step, episode)
            writer.add_scalar('eval/intercept_success_rate', eval_metrics.success_rate, episode)
            print(
                'episode:{}    epsilon:{:.4f}   Eval reward:{:.4f}   Reward/step:{:.4f}   Intercept Success:{:.2%}'.format(
                    episode,
                    float(np.mean([agent.e_greed for agent in agents])),
                    eval_metrics.mean_total_reward,
                    eval_metrics.mean_reward_per_step,
                    eval_metrics.success_rate,
                )
            )

        if episode % 100 == 0 and episode > 0:
            checkpoint_path = './models/DQNmodels/DDQNmodels3_23/DDQN_multi_episode{}.pth'.format(episode)
            checkpoint = {
                f'model_agent_{idx}': agent.model.state_dict() for idx, agent in enumerate(agents)
            }
            torch.save(checkpoint, checkpoint_path)

            success_rate = evaluate_checkpoint(checkpoint_path, validation_config)
            val_writer.add_scalar('intercept_success_rate', success_rate, episode)
            validation_results.append((episode, checkpoint_path, success_rate))
            validation_csv_path = save_validation_csv(val_log_dir, validation_results)
            print(
                'Validation after episode {}: intercept success rate {:.4f}'.format(
                    episode, success_rate
                )
            )
            print('Validation results saved to {}'.format(validation_csv_path))

    total_train_time = time.time() - start_time
    print('all used time {:.2f}s = {:.2f}h'.format(total_train_time, total_train_time / 3600))
    if validation_csv_path:
        print('Validation results saved to {}'.format(validation_csv_path))

    val_writer.close()


if __name__ == '__main__':
    main()
