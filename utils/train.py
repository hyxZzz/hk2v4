import argparse
import time
from dataclasses import dataclass
from typing import List, Sequence

import math
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


class ExponentialEpsilonScheduler:
    def __init__(
        self,
        epsilon_start: float,
        epsilon_end: float,
        decay_rate: float,
    ) -> None:
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.decay_rate = float(max(decay_rate, 0.0))
        self.total_env_steps = 0

    def get_epsilon(self) -> float:
        if self.decay_rate <= 0.0:
            return self.epsilon_start

        decayed = self.epsilon_end + (
            (self.epsilon_start - self.epsilon_end)
            * math.exp(-self.decay_rate * self.total_env_steps)
        )
        return max(self.epsilon_end, decayed)

    def step(self, steps: int = 1) -> None:
        self.total_env_steps += max(steps, 0)


@dataclass
class EvaluationMetrics:
    mean_total_reward: float
    mean_reward_per_step: float
    success_rate: float


def run_train_episode(
    agents: Sequence[MyDQNAgent],
    env,
    replay_buffers: Sequence[MyMemoryBuffer],
    memory_warmup_size: int,
    learn_freq: int,
    batch_size: int,
    train_loops: int,
    epsilon_scheduler=None,
):
    num_agents = len(agents)
    total_rewards = np.zeros(num_agents, dtype=np.float32)
    loss_sums = [0.0 for _ in range(num_agents)]
    loss_counts = [0 for _ in range(num_agents)]
    states, _, _ = env.reset()
    step = 0

    while True:
        step += 1
        if epsilon_scheduler is not None:
            epsilon = epsilon_scheduler.get_epsilon()
            for agent in agents:
                agent.e_greed = max(agent.min_epsilon, epsilon)

        actions = [agent.sample(states[i]) for i, agent in enumerate(agents)]
        next_states, rewards, done_flag, _ = env.step(actions)

        for idx in range(num_agents):
            experience = (states[idx], actions[idx], rewards[idx], next_states[idx], done_flag)
            replay_buffers[idx].add(experience)

        can_learn = step % learn_freq == 0 and all(
            buffer.size() > memory_warmup_size for buffer in replay_buffers
        )

        if can_learn:
            for _ in range(max(1, train_loops)):
                for idx, agent in enumerate(agents):
                    agent_buffer = replay_buffers[idx]
                    if agent_buffer.size() <= memory_warmup_size:
                        continue
                    experiences = agent_buffer.sample(batch_size)
                    if not experiences:
                        continue
                    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(
                        *experiences
                    )
                    loss = agent.learn(
                        batch_state, batch_action, batch_reward, batch_next_state, batch_done
                    )
                    loss_value = loss.detach().cpu().item() if torch.is_tensor(loss) else loss
                    loss_sums[idx] += float(loss_value)
                    loss_counts[idx] += 1

        total_rewards += np.array(rewards, dtype=np.float32)
        states = next_states

        if epsilon_scheduler is not None:
            epsilon_scheduler.step()

        if done_flag != -1:
            break

    mean_losses = [
        (loss_sums[idx] / loss_counts[idx]) if loss_counts[idx] > 0 else 0.0
        for idx in range(num_agents)
    ]

    return total_rewards, mean_losses, step


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

    parser.add_argument('--memory_size', type=int, default=160000, help='Size of replay memory')
    parser.add_argument('--memory_warmup_size', type=int, default=15000, help='Warmup size of replay memory')
    parser.add_argument('--learn_freq', type=int, default=1, help='Frequency of learning updates')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for training')
    parser.add_argument('--gamma', type=float, default=0.993, help='Discount factor')
    parser.add_argument(
        '--target_update_freq',
        type=int,
        default=40,
        help='Number of learning steps between target network updates',
    )
    parser.add_argument(
        '--target_update_tau',
        type=float,
        default=0.2,
        help='Soft update coefficient for target network (0 for hard update)',
    )
    parser.add_argument(
        '--train_loops',
        type=int,
        default=2,
        help='Number of gradient updates per learning step to better utilize the GPU',
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

    epsilon_start = 0.95
    epsilon_end = 0.02
    decay_ratio = 0.5
    expected_steps_per_episode = int(step_num * 0.9)
    decay_target_steps = max(
        1, int(expected_steps_per_episode * args.max_episode * decay_ratio)
    )
    target_epsilon = epsilon_end * 1.2
    ratio_numerator = max(target_epsilon - epsilon_end, 1e-6)
    ratio_denominator = max(epsilon_start - epsilon_end, 1e-6)
    decay_ratio_clamped = min(
        max(ratio_numerator / ratio_denominator, 1e-6), 0.999999
    )
    decay_rate = -math.log(decay_ratio_clamped) / decay_target_steps
    epsilon_scheduler = ExponentialEpsilonScheduler(
        epsilon_start, epsilon_end, decay_rate
    )

    agents: List[MyDQNAgent] = []
    for action_size in action_sizes:
        model = Double_DQN(state_size=state_size, action_size=action_size)
        agent = MyDQNAgent(
            model,
            action_size,
            gamma=args.gamma,
            lr=args.learning_rate,
            e_greed=epsilon_start,
            e_greed_decrement=0.0,
            min_epsilon=epsilon_end,
            update_target_steps=args.target_update_freq,
            soft_update_tau=args.target_update_tau,
            grad_clip=5.0,
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
            total_reward, train_losses, episode_steps = run_train_episode(
                agents,
                env,
                replay_buffers,
                args.memory_warmup_size,
                args.learn_freq,
                args.batch_size,
                args.train_loops,
                epsilon_scheduler,
            )
            mean_reward = float(np.mean(total_reward))
            mean_loss = float(np.mean(train_losses))
            writer.add_scalar('train/mean_reward', mean_reward, episode)
            writer.add_scalar('train/mean_loss', mean_loss, episode)
            current_epsilon = float(np.mean([agent.e_greed for agent in agents]))
            writer.add_scalar('train/epsilon', current_epsilon, episode)
            writer.add_scalar('train/episode_steps', episode_steps, episode)
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
