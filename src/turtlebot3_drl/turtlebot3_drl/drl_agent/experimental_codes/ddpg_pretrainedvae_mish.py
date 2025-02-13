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
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, 1)
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_out = torch.sigmoid(avg_out + max_out)
        x = x * channel_out
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_out
        return x

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class EnhancedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, transpose=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1) if not transpose else \
                    nn.ConvTranspose2d(in_channels, out_channels, 3, stride=stride, padding=1, output_padding=stride-1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.cbam = CBAM(out_channels)
        self.mish = Mish()
        
    def forward(self, x):
        out = self.mish(self.bn(self.conv(x)))
        out = self.cbam(out)
        return out

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder_blocks = nn.ModuleList([
            EnhancedBlock(1, 16, stride=2),
            EnhancedBlock(16, 32, stride=2),
            EnhancedBlock(32, 64, stride=2),
            EnhancedBlock(64, 128, stride=2)
        ])
        
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(128 * 5 * 10, latent_dim)
        self.fc_logvar = nn.Linear(128 * 5 * 10, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 128 * 5 * 10)
        
        self.decoder_blocks = nn.ModuleList([
            EnhancedBlock(128, 64, stride=2, transpose=True),
            EnhancedBlock(64, 32, stride=2, transpose=True),
            EnhancedBlock(32, 16, stride=2, transpose=True),
            EnhancedBlock(16, 8, stride=2, transpose=True)
        ])
        
        self.final_layer = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 128, 5, 10)
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)
        x = self.final_layer(x)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

# Modify Actor class
class Actor(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Actor, self).__init__(name)
        # --- define layers here ---

        # Load the first VAE model (for image1)
        self.vae1 = VAE(latent_dim=256)
        self.vae1.load_state_dict(torch.load('vae_model1.pth', map_location=torch.device('cpu')), strict=False)
        self.vae1.eval()

        # Load the second VAE model (for image2)
        self.vae2 = VAE(latent_dim=256)
        self.vae2.load_state_dict(torch.load('vae_model2.pth', map_location=torch.device('cpu')), strict=False)
        self.vae2.eval()

        # Fix encoder parameters
        for param in self.vae1.parameters():
            param.requires_grad = False
        for param in self.vae2.parameters():
            param.requires_grad = False

        # Define fully connected layers to pass through after combining additional state information
        conv_output_size = 256
        
        self.fc1 = nn.Linear(conv_output_size * 2 + 4, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        self.se = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 16),
            nn.ReLU(),
            nn.Linear(hidden_size // 16, hidden_size),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(p=0.2)
        
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, states, visualize=False):
        depth_image_size = 80 * 160

        if states.dim() == 2:
            # When batch is present
            li1 = states[:, :depth_image_size]
            li2 = states[:, depth_image_size:2 * depth_image_size]
            s1 = states[:, -4:]
        else:
            # When batch is not present
            li1 = states[:depth_image_size].view(1, -1)
            li2 = states[depth_image_size:2 * depth_image_size].view(1, -1)
            s1 = states[-4:].view(1, -1)

        #print("li1", li1, li1.min(), li1.max(), li1.shape, li1.dtype)


        darkness_factor = 1.0  # For 10% brightness


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


        # Extract latent vector through the first VAE encoder
        with torch.no_grad():
            mu1, _ = self.vae1.encode(li1_imaged)
            mu2, _ = self.vae2.encode(li2_imaged)

        # Combine the two latent vectors
        #latent_vector = torch.cat((mu1, mu2), dim=1)  # (batch_size, 96)

        # Combine additional state information
        #li_s1 = torch.cat((latent_vector, s1), dim=1)  # (batch_size, 100)

        # Pass through fully connected layers
        x = torch.cat((mu1, mu2, s1), dim=1)
        
        x = self.fc1(x)
        x = self.layer_norm1(x)
        x = F.relu(x)
        #x = x * self.se(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.layer_norm2(x)
        x = F.relu(x)
        #x = x * self.se(x)
        x = self.dropout(x)
        
        action = torch.tanh(self.fc3(x))
        
        if states.dim() == 2:
            action = action.view(x.size(0), -1)
        else:
            action = action.view(-1)
        
        return action

# Modify Critic class
class Critic(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Critic, self).__init__(name)
        # --- define layers here ---

        # --- Integrate VAE encoder ---
        # Load the first VAE model (for image1)
        self.vae1 = VAE(latent_dim=256)
        self.vae1.load_state_dict(torch.load('vae_model1.pth', map_location=torch.device('cpu')), strict=False)
        self.vae1.eval()

        # Load the second VAE model (for image2)
        self.vae2 = VAE(latent_dim=256)
        self.vae2.load_state_dict(torch.load('vae_model2.pth', map_location=torch.device('cpu')), strict=False)
        self.vae2.eval()

        # Fix encoder parameters to prevent training
        for param in self.vae1.parameters():
            param.requires_grad = False
        for param in self.vae2.parameters():
            param.requires_grad = False

        # --- Define Critic network layers ---
        conv_output_size = 256
        
        self.fc1 = nn.Linear(conv_output_size * 2 + 4, 256)
        self.layer_norm1 = nn.LayerNorm(256)
        
        self.fc2 = nn.Linear(action_size, 256)
        self.layer_norm2 = nn.LayerNorm(256)
        
        self.fc3 = nn.Linear(512, 512)
        self.layer_norm3 = nn.LayerNorm(512)
        
        self.fc4 = nn.Linear(512, 1)
        
        self.se = nn.Sequential(
            nn.Linear(256, 256 // 16),
            nn.ReLU(),
            nn.Linear(256 // 16, 256),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(p=0.2)
        
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, states, actions, visualize=False):
        depth_image_size = 80 * 160

        if states.dim() == 2:
            # When batch is present
            li1 = states[:, :depth_image_size]
            li2 = states[:, depth_image_size:2 * depth_image_size]
            s1 = states[:, -4:]
        else:
            # When batch is not present
            li1 = states[:depth_image_size].view(1, -1)
            li2 = states[depth_image_size:2 * depth_image_size].view(1, -1)
            s1 = states[-4:].view(1, -1)

        # Normalize data in the same way as during VAE training
        #li1_normalized = li1 / 255.0  # Scale pixel values to [0,1]

        # Convert to channel and image dimensions
        li1_imaged = li1.view(li1.size(0), 1, 80, 160)
        li2_imaged = li2.view(li2.size(0), 1, 80, 160)

        # Extract latent vector through the first VAE encoder
        with torch.no_grad():
            mu1, _ = self.vae1.encode(li1_imaged)
            mu2, _ = self.vae2.encode(li2_imaged)

        # Combine the two latent vectors
        #latent_vector = torch.cat((mu1, mu2), dim=1)  # (batch_size, 96)

        # Combine additional state information
        #li_s1 = torch.cat((latent_vector, s1), dim=1)  # (batch_size, 100)

        # Pass through Critic network layers
        xs = torch.cat((mu1, mu2, s1), dim=1)
        
        xs = self.fc1(xs)
        xs = self.layer_norm1(xs)
        xs = F.relu(xs)
        #xs = xs * self.se(xs)
        xs = self.dropout(xs)
        
        xa = self.fc2(actions)
        xa = self.layer_norm2(xa)
        xa = F.relu(xa)
        xa = self.dropout(xa)
        
        x = torch.cat((xs, xa), dim=1)
        
        x = self.fc3(x)
        x = self.layer_norm3(x)
        x = F.relu(x)
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