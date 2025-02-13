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

# 전체 VAE 클래스 정의 (인코더와 디코더 포함)
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        # 인코더 정의
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # (16, 80, 160)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # (16, 40, 80)

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # (32, 40, 80)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # (32, 20, 40)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # (64, 20, 40)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # (64, 10, 20)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),# (128, 10, 20)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # (128, 5, 10)

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),# (256, 5, 10)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                            # (256, 1, 1)
        )
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
    
        # 디코더 정의
        self.decoder_input = nn.Linear(latent_dim, 256)
        
        self.decoder = nn.Sequential(
            # Layer 1: ConvTranspose2d to (128,5,10)
            nn.ConvTranspose2d(256, 128, kernel_size=(5,10), stride=1, padding=0),  # (128,5,10)
            nn.ReLU(),
            
            # Layer 2: ConvTranspose2d with stride=2 to upsample to (64,10,20)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),    # (64,10,20)
            nn.ReLU(),
            
            # Layer 3: ConvTranspose2d with stride=2 to upsample to (32,20,40)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),     # (32,20,40)
            nn.ReLU(),
            
            # Layer 4: ConvTranspose2d with stride=2 to upsample to (16,40,80)
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),     # (16,40,80)
            nn.ReLU(),
            
            # Layer 5: ConvTranspose2d with stride=2 to upsample to (1,80,160)
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),      # (1,80,160)
            nn.Sigmoid()                                                      # Ensure output is in [0,1]
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
        x = self.decoder_input(z)             # (batch_size, 256)
        x = x.view(-1, 256, 1, 1)             # (batch_size, 256, 1, 1)
        x = self.decoder(x)                    # (batch_size, 1, 80, 160)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_groups=8):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out)
        return out

# Actor 클래스 수정
class Actor(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Actor, self).__init__(name)
        
        # Convolutional layers with residual blocks
        # For image1
        self.layer1_1 = ResidualBlock(1, 32, stride=2, num_groups=8)
        self.layer1_2 = ResidualBlock(32, 64, stride=2, num_groups=8)
        self.layer1_3 = ResidualBlock(64, 128, stride=2, num_groups=8)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # For image2
        self.layer2_1 = ResidualBlock(1, 32, stride=2, num_groups=8)
        self.layer2_2 = ResidualBlock(32, 64, stride=2, num_groups=8)
        self.layer2_3 = ResidualBlock(64, 128, stride=2, num_groups=8)
        
        # Calculate the size after convolutional layers
        conv_output_size = 128  # After adaptive pooling, the spatial dimensions are 1x1
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size * 2 + 4, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, 256)
        self.layer_norm2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, action_size)
        
        self.dropout = nn.Dropout(p=0.2)
        
        self.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, states, visualize=False):
        depth_image_size = 80 * 160
        
        if states.dim() == 2:
            # Batch processing
            li1 = states[:, :depth_image_size]
            li2 = states[:, depth_image_size:2 * depth_image_size]
            s1 = states[:, -4:]
        else:
            # Single sample
            li1 = states[:depth_image_size].view(1, -1)
            li2 = states[depth_image_size:2 * depth_image_size].view(1, -1)
            s1 = states[-4:].view(1, -1)
        
        # Reshape images
        li1 = li1.view(li1.size(0), 1, 80, 160)
        li2 = li2.view(li2.size(0), 1, 80, 160)
        
        # Process image1 through residual blocks
        x1 = self.layer1_1(li1)
        x1 = self.layer1_2(x1)
        x1 = self.layer1_3(x1)
        x1 = self.adaptive_pool(x1)
        x1 = x1.view(x1.size(0), -1)  # Flatten
        
        # Process image2 through residual blocks
        x2 = self.layer2_1(li2)
        x2 = self.layer2_2(x2)
        x2 = self.layer2_3(x2)
        x2 = self.adaptive_pool(x2)
        x2 = x2.view(x2.size(0), -1)  # Flatten
        
        # Concatenate features and state information
        x = torch.cat((x1, x2, s1), dim=1)
        
        # Fully connected layers
        x = F.relu(self.layer_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = self.dropout(x)
        action = torch.tanh(self.fc3(x))
        
        # Adjust output shape
        if states.dim() == 2:
            action = action.view(x.size(0), -1)
        else:
            action = action.view(-1)
        
        return action

# Critic 클래스 수정
class Critic(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Critic, self).__init__(name)
        
        # Convolutional layers with residual blocks
        # For image1
        self.layer1_1 = ResidualBlock(1, 32, stride=2, num_groups=8)
        self.layer1_2 = ResidualBlock(32, 64, stride=2, num_groups=8)
        self.layer1_3 = ResidualBlock(64, 128, stride=2, num_groups=8)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # For image2
        self.layer2_1 = ResidualBlock(1, 32, stride=2, num_groups=8)
        self.layer2_2 = ResidualBlock(32, 64, stride=2, num_groups=8)
        self.layer2_3 = ResidualBlock(64, 128, stride=2, num_groups=8)
        
        # Calculate the size after convolutional layers
        conv_output_size = 128  # After adaptive pooling, the spatial dimensions are 1x1
        
        # Fully connected layers
        # Note: Critic usually takes both state and action as input
        # Assuming action is concatenated after state features
        self.fc1 = nn.Linear(conv_output_size * 2 + 4, 256)
        self.layer_norm1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(action_size, 256)
        self.layer_norm2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(512, 512)
        self.layer_norm3 = nn.LayerNorm(512)
        self.fc4 = nn.Linear(512, 1)
        
        self.dropout = nn.Dropout(p=0.2)
        
        self.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, states, actions):
        depth_image_size = 80 * 160
        
        if states.dim() == 2:
            # Batch processing
            li1 = states[:, :depth_image_size]
            li2 = states[:, depth_image_size:2 * depth_image_size]
            s1 = states[:, -4:]
        else:
            # Single sample
            li1 = states[:depth_image_size].view(1, -1)
            li2 = states[depth_image_size:2 * depth_image_size].view(1, -1)
            s1 = states[-4:].view(1, -1)
        
        # Reshape images
        li1 = li1.view(li1.size(0), 1, 80, 160)
        li2 = li2.view(li2.size(0), 1, 80, 160)
        
        # Process image1 through residual blocks
        x1 = self.layer1_1(li1)
        x1 = self.layer1_2(x1)
        x1 = self.layer1_3(x1)
        x1 = self.adaptive_pool(x1)
        x1 = x1.view(x1.size(0), -1)  # Flatten
        
        # Process image2 through residual blocks
        x2 = self.layer2_1(li2)
        x2 = self.layer2_2(x2)
        x2 = self.layer2_3(x2)
        x2 = self.adaptive_pool(x2)
        x2 = x2.view(x2.size(0), -1)  # Flatten
        
        # Concatenate features, state information, and actions
        xs = torch.cat((x1, x2, s1), dim=1)
        
        # Fully connected layers
        xs = F.relu(self.layer_norm1(self.fc1(xs)))
        xs = self.dropout(xs)
        xa = F.relu(self.layer_norm2(self.fc2(actions)))
        xa = self.dropout(xa)
        x = torch.cat((xs, xa), dim=1)
        x = F.relu(self.layer_norm3(self.fc3(x)))
        x = self.dropout(x)
        value = self.fc4(x)
        
        return value


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