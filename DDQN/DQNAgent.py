import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Determine if CPU or GPU computation should be used
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 更新目标网络的操作函数，在MyDQNAgent.learn()函数中调用
def soft_update(target, source, tau=0.0):
    """Polyak averaging update for the target network."""

    tau = float(tau)
    if tau <= 0.0:
        target.load_state_dict(source.state_dict())
        return

    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.mul_(1.0 - tau)
            target_param.add_(tau * param)


class MyDQNAgent:

    def __init__(
        self,
        model,
        action_size,
        gamma=None,
        lr=None,
        e_greed=0.1,
        e_greed_decrement=0,
        min_epsilon=0.1,
        update_target_steps=15,
        soft_update_tau=0.0,
    ):

        self.action_size = action_size
        self.global_step = 0
        self.update_target_steps = max(1, int(update_target_steps))
        self.soft_update_tau = float(soft_update_tau)
        self.e_greed = e_greed  # ϵ-greedy中的ϵ
        self.e_greed_decrement = e_greed_decrement  # ϵ的动态更新因子（训练脚本负责调度）
        self.min_epsilon = min_epsilon
        self.model = model.to(device)
        self.target_model = copy.deepcopy(model).to(device)
        self.gamma = gamma  # 回报折扣因子
        self.lr = lr
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.optimizer = optim.Adam(lr=lr, params=self.model.parameters())

    def _update_target_model(self):
        soft_update(self.target_model, self.model, self.soft_update_tau)

    # 使用行为策略生成动作
    def sample(self, state):

        sample = np.random.random()  # [0.0, 1.0)
        if sample < self.e_greed:
            return np.random.randint(self.action_size)  # 返回[0, action_size)的整数，这里就是0或1

        return self.predict(state)

    # DQN网络做预测
    def predict(self, state):

        state = torch.tensor(state, dtype=torch.float32).to(device)
        # DQN网络做预测


        # '''此处将state置为全1数组看是否有影响'''
        # state = torch.ones(state.shape[0],  dtype=torch.float32)
        with torch.no_grad():
            pred_q = self.model(state)
        # 选取概率值最大的动作
        act = int(pred_q.argmax().item())

        # 在概率值前三大的三个动作中选
        # k = 5  # 前k个
        # b = pred_q.detach().numpy().argsort()[-3:][::-1]
        # act = np.random.choice(b, size=1)
        # act = act[0]


        return act

    # 更新DQN网络
    def learn(self, state, action, reward, next_state, terminal):
        """Update model with an episode data

        Args:
            state(np.float32): shape of (batch_size, state_size)
            act(np.int32): shape of (batch_size)
            reward(np.float32): shape of (batch_size)
            next_state(np.float32): shape of (batch_size, state_size)
            terminal(np.float32): shape of (batch_size)

        Returns:
            loss(float)
        """

        state = torch.as_tensor(np.array(state), dtype=torch.float32, device=device)
        action = torch.as_tensor(np.array(action), dtype=torch.int64, device=device).unsqueeze(-1)
        reward = torch.as_tensor(np.array(reward), dtype=torch.float32, device=device).unsqueeze(-1)
        next_state = torch.as_tensor(np.array(next_state), dtype=torch.float32, device=device)
        done = torch.as_tensor(
            (np.array(terminal) != -1).astype(np.float32),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(-1)

        # 1. DQN网络做正向传播
        pred_values = self.model(state)
        pred_value = pred_values.gather(1, action)

        # target Q
        with torch.no_grad():
            next_q_online = self.model(next_state)
            best_next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target_model(next_state)
            max_next_q = next_q_target.gather(1, best_next_actions)
            target = reward + (1.0 - done) * self.gamma * max_next_q

        # 4. TD 误差
        loss = self.mse_loss(pred_value, target)

        # 5. 更新DQN的参数
        # 梯度清零
        self.optimizer.zero_grad()
        # 反向计算梯度
        loss.backward()
        # 梯度更新
        self.optimizer.step()

        self.global_step += 1
        if self.global_step % self.update_target_steps == 0:
            self._update_target_model()

        return loss
