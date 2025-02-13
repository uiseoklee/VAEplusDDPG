import numpy as np
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
from turtlebot3_drl.drl_environment.reward import REWARD_FUNCTION
from ..common.settings import ENABLE_BACKWARD, ENABLE_STACKING

from ..common.ounoise import OUNoise
from ..drl_environment.drl_environment import NUM_SCAN_SAMPLES

from .off_policy_agent import OffPolicyAgent, Network
import matplotlib.pyplot as plt

# Encoder 클래스 수정
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                        
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        features = self.fc(x)
        return features

# Actor 클래스 수정
class Actor(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Actor, self).__init__(name)
        # --- define layers here ---

        self.state_size = state_size

        self.encoder1 = Encoder(latent_dim=23)
        self.encoder2 = Encoder(latent_dim=23)

        # 추가 상태 정보와 결합 후 통과할 완전 연결층 정의
        self.linear2 = nn.Linear(46 + 4, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, action_size)
        # --- define layers until here ---

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, states, visualize=False):
        depth_image_size = 80 * 160

        if states.dim() == 2:
            # 배치가 있는 경우
            li1 = states[:, :depth_image_size]
            li2 = states[:, depth_image_size:2 * depth_image_size]
            s1 = states[:, -4:]
        else:
            # 배치가 없는 경우
            li1 = states[:depth_image_size].view(1, -1)
            li2 = states[depth_image_size:2 * depth_image_size].view(1, -1)
            s1 = states[-4:].view(1, -1)

        print("li1", li1, li1.min(), li1.max(), li1.shape, li1.dtype)


        # VAE 학습 시와 동일한 방식으로 데이터 정규화
        #li1_normalized = li1 / 3.5  # 픽셀 값을 [0,1]로 스케일링
        # print("li1_normalized.size()", li1_normalized.size())

        # 채널 및 이미지 차원으로 변환
        li1_normalized = li1.view(li1.size(0), 1, 80, 160)  # (batch_size, 1, 80, 160)
        
        
        batch_size = li1.size(0)
        # Convert li1 into a NumPy array if it's a tensor
        if isinstance(li1_normalized, torch.Tensor):
            li1_view = li1_normalized.detach().cpu().numpy()

        # Check the shape of li1_normalized and reshape if necessary
        #li1 = li1.reshape(batch_size, 1, 80, 160)
        if li1_view.ndim == 4:
            # Assuming the shape is (batch_size, channels, height, width)
            # Select the first image in the batch and the first channel
            li1_view = li1_view[0, 0, :, :]

        # Plot the image
        plt.imshow(li1_view, cmap='gray')
        plt.colorbar()
        plt.title('li1_normalized')
        plt.show()
        

        li2_normalized = li2.view(li2.size(0), 1, 80, 160)

        # Conv2d 인코더를 통해 잠재 벡터 추출
        encoded1 = self.encoder1(li1_normalized)
        encoded2 = self.encoder2(li2_normalized)

        # 잠재 벡터와 추가 상태 정보 결합
        li_s1 = torch.cat((encoded1, encoded2), dim=1)

        li_s1 = torch.cat((li_s1, s1), dim=1)

        # 완전 연결층 통과
        li_s2 = torch.relu(self.linear2(li_s1))  # (batch_size, hidden_size)
        li_s3 = torch.relu(self.linear3(li_s2))  # (batch_size, hidden_size)
        action = torch.tanh(self.linear4(li_s3))  # (batch_size, action_size)

        # 출력 형태 조정
        if states.dim() == 2:
            action = action.view(li_s3.size(0), -1)  # (batch_size, action_size)
        else:
            action = action.view(-1)  # (action_size,)

        # -- define layers to visualize here (optional) ---
        if visualize and self.visual:
            self.visual.update_layers(states, action, [li_s2, li_s3], [self.linear2.bias, self.linear3.bias])
        # -- define layers to visualize until here ---
        return action

# Critic 클래스 수정
class Critic(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Critic, self).__init__(name)
        # --- define layers here ---

        # Combined strategies for aggressive reduction
        self.encoder1 = Encoder(latent_dim=23)
        self.encoder2 = Encoder(latent_dim=23)

        # --- Critic 네트워크 레이어 정의 ---
        # 입력 크기를 잠재 벡터(46) + s1(4)로 조정
        self.l1 = nn.Linear(46 + 4, int(hidden_size / 2))
        self.l2 = nn.Linear(action_size, int(hidden_size / 2))
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, 1)
        # --- 레이어 정의 끝 ---

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, states, actions, visualize=False):
        depth_image_size = 80 * 160

        if states.dim() == 2:
            # 배치가 있는 경우
            li1 = states[:, :depth_image_size]
            li2 = states[:, depth_image_size:2 * depth_image_size]
            s1 = states[:, -4:]
        else:
            # 배치가 없는 경우
            li1 = states[:depth_image_size].view(1, -1)
            li2 = states[depth_image_size:2 * depth_image_size].view(1, -1)
            s1 = states[-4:].view(1, -1)

        # VAE 학습 시와 동일한 방식으로 데이터 정규화
        #li1_normalized = li1 / 255.0  # 픽셀 값을 [0,1]로 스케일링

        # 채널 및 이미지 차원으로 변환
        li1_normalized = li1.view(li1.size(0), 1, 80, 160)
        li2_normalized = li2.view(li2.size(0), 1, 80, 160)

        # Conv2d 인코더를 통해 잠재 벡터 추출
        encoded1 = self.encoder1(li1_normalized)
        encoded2 = self.encoder2(li2_normalized)

        # 잠재 벡터와 추가 상태 정보 결합
        li_s1 = torch.cat((encoded1, encoded2), dim=1)
        li_s1 = torch.cat((li_s1, s1), dim=1)

        # Critic 네트워크 레이어 통과
        xs = torch.relu(self.l1(li_s1))       # (batch_size, hidden_size / 2)
        xa = torch.relu(self.l2(actions))     # (batch_size, hidden_size / 2)
        x = torch.cat((xs, xa), dim=1)        # (batch_size, hidden_size)
        x = torch.relu(self.l3(x))            # (batch_size, hidden_size)
        x = self.l4(x)                         # (batch_size, 1)

        # -- 시각화를 위한 레이어 업데이트 (선택 사항) ---
        if visualize and self.visual:
            self.visual.update_layers(states, x, [xs, xa, x], [self.l1.bias, self.l2.bias, self.l3.bias])
        # -- 시각화를 위한 레이어 업데이트 끝 ---
        return x


class DDPG(OffPolicyAgent):
    def __init__(self, device, sim_speed):
        super().__init__(device, sim_speed)

        self.noise = OUNoise(action_space=self.action_size, max_sigma=0.1, min_sigma=0.1, decay_period=8000000)

        self.actor = self.create_network(Actor, 'actor')
        self.actor_target = self.create_network(Actor, 'target_actor')
        self.actor_optimizer = self.create_optimizer(self.actor)

        self.critic = self.create_network(Critic, 'critic')
        self.critic_target = self.create_network(Critic, 'target_critic')
        self.critic_optimizer = self.create_optimizer(self.critic)

        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

    def get_action(self, state, is_training, step, visualize=False):
        state = torch.from_numpy(np.asarray(state, np.float32)).to(self.device)
        action = self.actor(state, visualize)
        if is_training:
            noise = torch.from_numpy(copy.deepcopy(self.noise.get_noise(step))).to(self.device)
            action = torch.clamp(torch.add(action, noise), -1.0, 1.0)
        return action.detach().cpu().data.numpy().tolist()

    def get_action_random(self):
        return [np.clip(np.random.uniform(-1.0, 1.0), -1.0, 1.0)] * self.action_size

    def train(self, state, action, reward, state_next, done):
        # optimize critic
        action_next = self.actor_target(state_next)
        Q_next = self.critic_target(state_next, action_next)
        Q_target = reward + (1 - done) * self.discount_factor * Q_next
        Q = self.critic(state, action)

        loss_critic = self.loss_function(Q, Q_target)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2.0, norm_type=2)
        self.critic_optimizer.step()

        pred_a_sample = self.actor(state)
        loss_actor = -1 * (self.critic(state, pred_a_sample)).mean()

        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0, norm_type=2)
        self.actor_optimizer.step()

        # Soft update all target networks
        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)

        return [loss_critic.mean().detach().cpu(), loss_actor.mean().detach().cpu()]