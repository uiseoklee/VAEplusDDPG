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


class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        return attn @ v


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return torch.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // 16),
            nn.ReLU(),
            nn.Linear(in_planes // 16, in_planes)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return torch.sigmoid(out).view(x.size(0), x.size(1), 1, 1)

"""
class FPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_channels):
        super(FPNBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn = nn.GroupNorm(8, out_channels)
        if up_channels != out_channels:
            self.up_conv = nn.Conv2d(up_channels, out_channels, kernel_size=1)
        else:
            self.up_conv = nn.Identity()

    def forward(self, x, up=None):
        x = self.conv1(x)
        if up is not None:
            up = self.up_conv(up)
            up = F.interpolate(up, size=x.shape[-2:], mode='nearest')
            x = x + up
        x = self.conv2(x)
        x = self.gn(x)
        return F.relu(x)
"""
        

class EnhancedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.gn3 = nn.GroupNorm(8, out_channels)
        
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()
        
        self.dropout = nn.Dropout2d(0.1)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.GroupNorm(8, out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = F.relu(self.gn1(self.conv1(x)))
        out = F.relu(self.gn2(self.conv2(out)))
        out = self.gn3(self.conv3(out))
        
        out = self.ca(out) * out
        out = self.sa(out) * out
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = F.relu(out)
        return self.dropout(out)

class Network(nn.Module):
    def __init__(self, name):
        super(Network, self).__init__()
        self.name = name

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class Actor(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super().__init__(name)
        
        # Enhanced feature extraction
        self.encoder1 = nn.ModuleList([
            EnhancedResidualBlock(1, 32, stride=2),
            EnhancedResidualBlock(32, 64, stride=2),
            EnhancedResidualBlock(64, 128, stride=2)
        ])
        
        self.encoder2 = nn.ModuleList([
            EnhancedResidualBlock(1, 32, stride=2),
            EnhancedResidualBlock(32, 64, stride=2),
            EnhancedResidualBlock(64, 128, stride=2)
        ])
        
        self.cross_attention = CrossAttention(128)
        
        # FPN-style feature fusion
        #self.fpn = nn.ModuleList([
        #    FPNBlock(in_channels=128, out_channels=128, up_channels=128),
        #    FPNBlock(in_channels=64, out_channels=128, up_channels=128),
        #    FPNBlock(in_channels=32, out_channels=128, up_channels=128)
        #])
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Advanced decision layers
        self.fc1 = nn.Linear(256 + 4, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln2 = nn.LayerNorm(hidden_size // 2)
        
        self.fc3 = nn.Linear(hidden_size // 2, action_size)
        
        self.se = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 16),
            nn.ReLU(),
            nn.Linear(hidden_size // 16, hidden_size),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(0.2)
        
        self.apply(self.init_weights)
    
    def forward(self, states, visualize=False):
        depth_image_size = 80 * 160
        
        if states.dim() == 2:
            li1 = states[:, :depth_image_size]
            li2 = states[:, depth_image_size:2 * depth_image_size]
            s1 = states[:, -4:]
        else:
            li1 = states[:depth_image_size].view(1, -1)
            li2 = states[depth_image_size:2 * depth_image_size].view(1, -1)
            s1 = states[-4:].view(1, -1)
        
        li1 = li1.view(li1.size(0), 1, 80, 160)
        li2 = li2.view(li2.size(0), 1, 80, 160)
        
        # Process first image
        features1 = []
        x1 = li1
        for block in self.encoder1:
            x1 = block(x1)
            features1.append(x1)
        
        # Process second image
        features2 = []
        x2 = li2
        for block in self.encoder2:
            x2 = block(x2)
            features2.append(x2)
        
        
        # Cross attention between final features
        b, c, h, w = x1.shape
        x1_flat = x1.view(b, c, -1).transpose(-2, -1)
        x2_flat = x2.view(b, c, -1).transpose(-2, -1)
        
        x1_attended = self.cross_attention(x1_flat, x2_flat)
        x2_attended = self.cross_attention(x2_flat, x1_flat)
        
        x1 = x1_attended.transpose(-2, -1).view(b, c, h, w)
        x2 = x2_attended.transpose(-2, -1).view(b, c, h, w)
        

        # FPN feature fusion
        #for fpn_block, f1, f2 in zip(self.fpn, reversed(features1), reversed(features2)):
        #    x1 = fpn_block(f1, x1)
        #    x2 = fpn_block(f2, x2)
        
        # Global pooling
        x1 = self.adaptive_pool(x1).view(x1.size(0), -1)
        x2 = self.adaptive_pool(x2).view(x2.size(0), -1)
        
        # Concatenate features with state information
        x = torch.cat((x1, x2, s1), dim=1)
        
        # Final decision layers
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = x * self.se(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        action = torch.tanh(self.fc3(x))
        
        if states.dim() == 2:
            action = action.view(x.size(0), -1)
        else:
            action = action.view(-1)
        
        return action

class Critic(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super().__init__(name)
        
        # Similar structure to Actor for state processing
        self.encoder1 = nn.ModuleList([
            EnhancedResidualBlock(1, 32, stride=2),
            EnhancedResidualBlock(32, 64, stride=2),
            EnhancedResidualBlock(64, 128, stride=2)
        ])
        
        self.encoder2 = nn.ModuleList([
            EnhancedResidualBlock(1, 32, stride=2),
            EnhancedResidualBlock(32, 64, stride=2),
            EnhancedResidualBlock(64, 128, stride=2)
        ])
        
        self.cross_attention = CrossAttention(128)
        
        #self.fpn = nn.ModuleList([
        #    FPNBlock(in_channels=128, out_channels=128, up_channels=128),
        #    FPNBlock(in_channels=64, out_channels=128, up_channels=128),
        #    FPNBlock(in_channels=32, out_channels=128, up_channels=128)
        #])
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Action processing
        self.action_fc = nn.Linear(action_size, 256)
        self.action_ln = nn.LayerNorm(256)
        
        # Combined processing
        self.fc1 = nn.Linear(256 + 256 + 4, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln2 = nn.LayerNorm(hidden_size // 2)
        
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        
        self.se = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 16),
            nn.ReLU(),
            nn.Linear(hidden_size // 16, hidden_size),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(0.2)
        
        self.apply(self.init_weights)
    
    def forward(self, states, actions):
        depth_image_size = 80 * 160
        
        if states.dim() == 2:
            li1 = states[:, :depth_image_size]
            li2 = states[:, depth_image_size:2 * depth_image_size]
            s1 = states[:, -4:]
        else:
            li1 = states[:depth_image_size].view(1, -1)
            li2 = states[depth_image_size:2 * depth_image_size].view(1, -1)
            s1 = states[-4:].view(1, -1)
        
        li1 = li1.view(li1.size(0), 1, 80, 160)
        li2 = li2.view(li2.size(0), 1, 80, 160)
        
        # Process images similar to Actor
        features1 = []
        x1 = li1
        for block in self.encoder1:
            x1 = block(x1)
            features1.append(x1)
        
        features2 = []
        x2 = li2
        for block in self.encoder2:
            x2 = block(x2)
            features2.append(x2)
        
        
        # Cross attention
        b, c, h, w = x1.shape
        x1_flat = x1.view(b, c, -1).transpose(-2, -1)
        x2_flat = x2.view(b, c, -1).transpose(-2, -1)
        
        x1_attended = self.cross_attention(x1_flat, x2_flat)
        x2_attended = self.cross_attention(x2_flat, x1_flat)
        
        x1 = x1_attended.transpose(-2, -1).view(b, c, h, w)
        x2 = x2_attended.transpose(-2, -1).view(b, c, h, w)
        

        # FPN feature fusion
        #for fpn_block, f1, f2 in zip(self.fpn, reversed(features1), reversed(features2)):
        #    x1 = fpn_block(f1, x1)
        #    x2 = fpn_block(f2, x2)
        
        # Global pooling
        x1 = self.adaptive_pool(x1).view(x1.size(0), -1)
        x2 = self.adaptive_pool(x2).view(x2.size(0), -1)
        
        # Process state features
        state_features = torch.cat((x1, x2, s1), dim=1)
        
        # Process actions
        action_features = F.relu(self.action_ln(self.action_fc(actions)))
        
        # Combine all features
        x = torch.cat((state_features, action_features), dim=1)
        
        # Final decision layers
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = x * self.se(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        value = self.fc3(x)
        
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