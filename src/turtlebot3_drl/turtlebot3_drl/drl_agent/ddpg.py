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

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, 1)
        )
        
        # Spatial Attention
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Channel Attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_out = torch.sigmoid(avg_out + max_out)
        x = x * channel_out
        
        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_out
        return x

class EnhancedLightFPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.gn2 = nn.GroupNorm(8, out_channels)
        self.cbam = CBAM(out_channels)
        #self.se = nn.Sequential(
        #    nn.AdaptiveAvgPool2d(1),
        #    nn.Conv2d(out_channels, out_channels // 16, 1),
        #    nn.ReLU(),
        #    nn.Conv2d(out_channels // 16, out_channels, 1),
        #    nn.Sigmoid()
        #)
        
        if up_channels != out_channels:
            self.up_conv = nn.Conv2d(up_channels, out_channels, kernel_size=1)
        else:
            self.up_conv = nn.Identity()

        # 다운샘플링 레이어 추가
        if in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.downsample = nn.Identity()            
    
    def forward(self, x, up=None):
        identity = x
        # identity에 다운샘플링 적용
        identity = self.downsample(identity)

        x = self.conv1(x)
        x = F.gelu(self.gn1(x))
        x = self.conv2(x)
        x = F.gelu(self.gn2(x))
        
        if up is not None:
            up = self.up_conv(up)
            up = F.interpolate(up, size=x.shape[-2:], mode='nearest')
            x = x + up
        
        x = self.cbam(x)
        #x = x * self.se(x)
        
        x = x + identity
        return x

class EnhancedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_channels)
        
        self.cbam = CBAM(out_channels)
        #self.se = nn.Sequential(
        #    nn.AdaptiveAvgPool2d(1),
        #    nn.Conv2d(out_channels, out_channels // 16, 1),
        #    nn.ReLU(),
        #    nn.Conv2d(out_channels // 16, out_channels, 1),
        #    nn.Sigmoid()
        #)
        
        self.dropout = nn.Dropout2d(0.1)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.GroupNorm(8, out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = F.gelu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        
        out = self.cbam(out)
        #out = out * self.se(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = F.gelu(out)
        return self.dropout(out)

class Network(nn.Module):
    def __init__(self, name):
        super(Network, self).__init__()
        self.name = name

    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

class Actor(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super().__init__(name)
        
        # Encoder layers
        self.encoder = nn.ModuleList([
            EnhancedResidualBlock(1, 32, stride=2),
            EnhancedResidualBlock(32, 64, stride=2),
            EnhancedResidualBlock(64, 128, stride=2)
        ])
        
        # Light FPN layers
        self.fpn = nn.ModuleList([
            EnhancedLightFPNBlock(in_channels=128, out_channels=128, up_channels=128),
            EnhancedLightFPNBlock(in_channels=64, out_channels=128, up_channels=128),
            EnhancedLightFPNBlock(in_channels=32, out_channels=128, up_channels=128)
        ])
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        # Decision layers
        self.fc1 = nn.Linear(256 + 4, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln2 = nn.LayerNorm(hidden_size // 2)
        
        self.fc3 = nn.Linear(hidden_size // 2, action_size)
        
        self.dropout = nn.Dropout(0.1)
        
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

        darkness_factor = 0.3  # For 10% brightness


        """
        # Normalize the image data to [0, 1]
        li1_normalized_forplt = li1 / 8.0  # Assuming li1 has values in [0, 8.0]

        # Apply darkness_factor to adjust brightness
        li1_darkened = li1_normalized_forplt * darkness_factor

        # Ensure the pixel values remain in [0, 1]
        li1_darkened = li1_darkened.clamp(0, 1)

        # Reshape to match network input dimensions
        li1_darkened = li1_darkened.view(li1.size(0), 1, 80, 160)  # (batch_size, 1, 80, 160)

        # Convert li1_normalized into a NumPy array if it's a tensor
        if isinstance(li1_darkened, torch.Tensor):
            li1_view = li1_darkened.detach().cpu().numpy()

        # Check the shape of li1_view and reshape if necessary
        if li1_view.ndim == 4:
            # Assuming the shape is (batch_size, channels, height, width)
            # Select the first image in the batch and the first channel
            li1_view = li1_view[0, 0, :, :]

        # Plot the image with fixed vmin and vmax to reflect darkness_factor
        plt.imshow(li1_view, cmap='gray', vmin=0, vmax=1)
        plt.colorbar()
        plt.title(f'li1_normalized with darkness_factor {darkness_factor} applied')
        plt.show()
        """

        li1_imaged = li1.view(li1.size(0), 1, 80, 160)
        li2_imaged = li2.view(li2.size(0), 1, 80, 160)

        # Apply darkness_factor
        li1_imaged = li1_imaged * darkness_factor
        li2_imaged = li2_imaged * darkness_factor 
        
        # Process first image
        features1 = []
        x1 = li1_imaged
        for encoder_layer in self.encoder:
            x1 = encoder_layer(x1)
            features1.append(x1)
        
        # Process second image
        features2 = []
        x2 = li2_imaged
        for encoder_layer in self.encoder:
            x2 = encoder_layer(x2)
            features2.append(x2)
        
        # FPN feature fusion
        for fpn_block, f1, f2 in zip(self.fpn, reversed(features1), reversed(features2)):
            x1 = fpn_block(f1, x1)
            x2 = fpn_block(f2, x2)
        
        x1 = self.adaptive_pool(x1)
        x1 = self.flatten(x1)
        
        x2 = self.adaptive_pool(x2)
        x2 = self.flatten(x2)
        
        # Combine features
        x = torch.cat((x1, x2, s1), dim=1)
        
        # Decision layers with residual connection
        #identity = x
        x = F.gelu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.gelu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        #x = x + identity[:, :x.size(1)]  # Residual connection
        
        action = torch.tanh(self.fc3(x))
        
        if states.dim() == 2:
            action = action.view(x.size(0), -1)
        else:
            action = action.view(-1)
        
        return action

class Critic(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super().__init__(name)
        
        # Encoder layers
        self.encoder = nn.ModuleList([
            EnhancedResidualBlock(1, 32, stride=2),
            EnhancedResidualBlock(32, 64, stride=2),
            EnhancedResidualBlock(64, 128, stride=2)
        ])
        
        # Light FPN layers
        self.fpn = nn.ModuleList([
            EnhancedLightFPNBlock(in_channels=128, out_channels=128, up_channels=128),
            EnhancedLightFPNBlock(in_channels=64, out_channels=128, up_channels=128),
            EnhancedLightFPNBlock(in_channels=32, out_channels=128, up_channels=128)
        ])
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        # Action processing
        self.action_fc = nn.Linear(action_size, hidden_size // 2)
        self.action_ln = nn.LayerNorm(hidden_size // 2)
        
        # Combined processing
        self.fc1 = nn.Linear(256 + hidden_size // 2 + 4, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln2 = nn.LayerNorm(hidden_size // 2)
        
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        
        self.dropout = nn.Dropout(0.1)
        
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
        
        li1_imaged = li1.view(li1.size(0), 1, 80, 160)
        li2_imaged = li2.view(li2.size(0), 1, 80, 160)



        darkness_factor = 0.3 # For 10% brightness

        # Apply darkness_factor
        li1_imaged = li1_imaged * darkness_factor
        li2_imaged = li2_imaged * darkness_factor



        # Process first image
        features1 = []
        x1 = li1_imaged
        for encoder_layer in self.encoder:
            x1 = encoder_layer(x1)
            features1.append(x1)
        
        # Process second image
        features2 = []
        x2 = li2_imaged
        for encoder_layer in self.encoder:
            x2 = encoder_layer(x2)
            features2.append(x2)
        
        # FPN feature fusion
        for fpn_block, f1, f2 in zip(self.fpn, reversed(features1), reversed(features2)):
            x1 = fpn_block(f1, x1)
            x2 = fpn_block(f2, x2)
        
        x1 = self.adaptive_pool(x1)
        x1 = self.flatten(x1)

        x2 = self.adaptive_pool(x2)
        x2 = self.flatten(x2)
        
        # Process state features
        state_features = torch.cat((x1, x2, s1), dim=1)
        
        # Process actions
        action_features = F.gelu(self.action_ln(self.action_fc(actions)))
        
        # Combine features
        x = torch.cat((state_features, action_features), dim=1)
        
        # Decision layers with residual connection
        #identity = x
        x = F.gelu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.gelu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        #x = x + identity[:, :x.size(1)]  # Residual connection
        
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