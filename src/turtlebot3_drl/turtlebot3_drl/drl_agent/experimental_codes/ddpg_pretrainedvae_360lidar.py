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

LINEAR = 0
ANGULAR = 1

# Reference for network structure: https://arxiv.org/pdf/2102.10711.pdf
# https://github.com/hanlinniu/turtlebot3_ddpg_collision_avoidance/blob/main/turtlebot_ddpg/scripts/original_ddpg/ddpg_network_turtlebot3_original_ddpg.py

# VAE 클래스 정의
class VAE(nn.Module):
    def __init__(self, latent_dim=46):
        super(VAE, self).__init__()
        # 인코더 정의
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 360 -> 180
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 180 -> 90
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 90 -> 45
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
            # MaxPool1d 제거하여 시퀀스 길이 45 유지
        )
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(256 * 45, latent_dim)
        self.fc_logvar = nn.Linear(256 * 45, latent_dim)

        # 디코더 정의
        self.decoder_input = nn.Linear(latent_dim, 256 * 45)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 45 -> 90
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 90 -> 180
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 180 -> 360
            nn.Conv1d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # 출력 값을 [0,1]로 제한
        )

    def encode(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # logvar 값 제한
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 256, 45)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

# Actor 클래스 수정
class Actor(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Actor, self).__init__(name)
        # --- define layers here ---

        self.state_size = state_size

        # VAE 모델 로드
        self.vae = VAE(latent_dim=46)
        self.vae.load_state_dict(torch.load('vae_model.pth', map_location=torch.device('cpu')))
        self.vae.eval()  # 평가 모드로 설정

        # 인코더 부분만 사용
        self.encoder = self.vae.encoder
        self.fc_mu = self.vae.fc_mu

        # 인코더의 파라미터를 고정하여 학습되지 않도록 설정
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.fc_mu.parameters():
            param.requires_grad = False

        # 완전 연결층 정의
        self.linear2 = nn.Linear(46 + 4, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, action_size)
        # --- define layers until here ---

        self.apply(super().init_weights)

    def forward(self, states, visualize=False):
        if states.dim() == 2:
            # 배치가 있는 경우
            li1 = states[:, :states.shape[1] - 4]  # (batch_size, 360)
            s1 = states[:, states.shape[1] - 4:]   # (batch_size, 4)
        else:
            # 배치가 없는 경우
            li1 = states[:states.shape[0] - 4].view(1, -1)  # (1, 360)
            s1 = states[states.shape[0] - 4:].view(1, -1)   # (1, 4)
            
        #print("li1", li1, li1.min(), li1.max(), li1.shape)

        # VAE 학습 시와 동일한 방식으로 데이터 정규화
        data_min = li1.min(dim=1, keepdim=True)[0]
        data_max = li1.max(dim=1, keepdim=True)[0]
        li1_normalized = (li1 - data_min) / (data_max - data_min + 1e-8)

        # 채널 차원 추가
        li1_normalized = li1_normalized.unsqueeze(1)  # (batch_size, 1, 360)

        # VAE 인코더를 통해 잠재 벡터 추출
        with torch.no_grad():
            encoded = self.encoder(li1_normalized)
            encoded = encoded.view(encoded.size(0), -1)
            mu = self.fc_mu(encoded)  # (batch_size, 46)

        # 잠재 벡터와 추가 상태 정보 결합
        li_s1 = torch.cat((mu, s1), dim=1)  # (batch_size, 50)

        # 완전 연결층 통과
        li_s2 = torch.relu(self.linear2(li_s1))
        li_s3 = torch.relu(self.linear3(li_s2))
        action = torch.tanh(self.linear4(li_s3))

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


class Critic(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Critic, self).__init__(name)
        #hidden_size = hidden_size * 10
        # --- define layers here ---
        
        self.l1 = nn.Linear(state_size, int(hidden_size / 2))
        self.l2 = nn.Linear(action_size, int(hidden_size / 2))
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, 1)
        # --- define layers until here ---

        self.apply(super().init_weights)

    def forward(self, states, actions):
        # --- define forward pass here ---
        
        xs = torch.relu(self.l1(states))
        xa = torch.relu(self.l2(actions))
        x = torch.cat((xs, xa), dim=1)
        x = torch.relu(self.l3(x))
        x = self.l4(x)
        

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
