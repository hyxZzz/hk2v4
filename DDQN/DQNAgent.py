import copy
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def soft_update(target, source, tau=0):
    target.load_state_dict(source.state_dict())


class MyDQNAgent:
    def __init__(
        self,
        model,
        action_size_each,
        num_agents: int = 2,
        gamma: Optional[float] = None,
        lr: Optional[float] = None,
        e_greed: float = 0.1,
        e_greed_decrement: float = 0.0,
        update_target_steps: int = 15,
    ):
        self.action_size_each = action_size_each
        self.num_agents = num_agents
        self.global_step = 0
        self.update_target_steps = max(1, int(update_target_steps))
        self.e_greed = e_greed
        self.e_greed_decrement = e_greed_decrement
        self.model = model.to(device)
        self.target_model = copy.deepcopy(model).to(device)
        self.gamma = gamma
        self.lr = lr
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.optimizer = optim.Adam(lr=lr, params=self.model.parameters())

    def _update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def sample(self, state) -> Tuple[int, ...]:
        greedy_actions = self.predict(state)
        actions = []
        for head in range(self.num_agents):
            if np.random.random() < self.e_greed:
                actions.append(int(np.random.randint(self.action_size_each)))
            else:
                actions.append(greedy_actions[head])
        self.e_greed = max(0.1, self.e_greed - self.e_greed_decrement)
        return tuple(actions)

    def predict(self, state) -> Tuple[int, ...]:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        actions = [int(head_q.argmax(dim=-1).item()) for head_q in q_values]
        return tuple(actions)

    def learn(
        self,
        state: Sequence,
        action: Sequence,
        reward: Sequence,
        next_state: Sequence,
        terminal: Sequence,
        attribution: Optional[Sequence] = None,
    ):
        state_tensor = torch.as_tensor(np.array(state), dtype=torch.float32, device=device)
        action_tensor = torch.as_tensor(np.array(action), dtype=torch.int64, device=device)
        reward_tensor = torch.as_tensor(np.array(reward), dtype=torch.float32, device=device).unsqueeze(-1)
        next_state_tensor = torch.as_tensor(np.array(next_state), dtype=torch.float32, device=device)
        done_tensor = torch.as_tensor(
            (np.array(terminal) != -1).astype(np.float32), dtype=torch.float32, device=device
        ).unsqueeze(-1)

        if attribution is not None:
            attr_tensor = torch.as_tensor(np.array(attribution), dtype=torch.float32, device=device)
        else:
            attr_tensor = torch.zeros((state_tensor.shape[0], self.num_agents), dtype=torch.float32, device=device)

        current_qs = self.model(state_tensor)
        next_online = self.model(next_state_tensor)
        next_target = self.target_model(next_state_tensor)

        losses = []
        for agent_idx in range(self.num_agents):
            q_pred = current_qs[agent_idx].gather(1, action_tensor[:, agent_idx].unsqueeze(-1))
            best_next = next_online[agent_idx].argmax(dim=1, keepdim=True)
            q_next = next_target[agent_idx].gather(1, best_next)
            target = reward_tensor + attr_tensor[:, agent_idx].unsqueeze(-1)
            target = target + (1.0 - done_tensor) * self.gamma * q_next
            losses.append(self.mse_loss(q_pred, target))

        loss = sum(losses) / self.num_agents

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.global_step += 1
        if self.global_step % self.update_target_steps == 0:
            self._update_target_model()

        return loss
