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

# CBAM Module
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

# Enhanced Light FPN Block
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

        # Downsample layer
        if in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.downsample = nn.Identity()            
    
    def forward(self, x, up=None):
        identity = x
        # Apply downsample to identity
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

# Enhanced Residual Block
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

# Enhanced ConvTranspose2d Block with Attention
class EnhancedConvTranspose2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=stride, padding=1, output_padding=1)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1)
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
        
        self.upsample = None
        if stride != 1 or in_channels != out_channels:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, output_padding=1),
                nn.GroupNorm(8, out_channels)
            )
    
    def forward(self, x):
        identity = x
        out = F.gelu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        
        out = self.cbam(out)
        #out = out * self.se(out)
        
        if self.upsample is not None:
            identity = self.upsample(x)
            
        out += identity
        out = F.gelu(out)
        return self.dropout(out)

# VAE Class with Enhanced Blocks
class Conv2dVAE(nn.Module):
    def __init__(self, latent_dim):
        super(Conv2dVAE, self).__init__()
        
        # Encoder
        self.encoder_blocks = nn.ModuleList([
            EnhancedResidualBlock(1, 32, stride=2),    # Output: (batch_size, 32, 40, 80)
            EnhancedResidualBlock(32, 64, stride=2),   # Output: (batch_size, 64, 20, 40)
            EnhancedResidualBlock(64, 128, stride=2)   # Output: (batch_size, 128, 10, 20)
        ])
        
        # FPN Layers for Encoder
        self.fpn_encoder = nn.ModuleList([
            EnhancedLightFPNBlock(in_channels=128, out_channels=128, up_channels=128),
            EnhancedLightFPNBlock(in_channels=64, out_channels=128, up_channels=128),
            EnhancedLightFPNBlock(in_channels=32, out_channels=128, up_channels=128)
        ])
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output: (batch_size, 128, 1, 1)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 128 * 10 * 20)
        
        self.decoder_blocks = nn.ModuleList([
            EnhancedConvTranspose2dBlock(128, 128, stride=2),    # Output: (batch_size, 128, 20, 40)
            EnhancedConvTranspose2dBlock(128, 64, stride=2),     # Output: (batch_size, 64, 40, 80)
            EnhancedConvTranspose2dBlock(64, 32, stride=2),      # Output: (batch_size, 32, 80, 160)
        ])
        
        self.final_layer = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1),  # Output: (batch_size, 1, 80, 160)
            nn.Sigmoid()
        )
    
    def encode(self, x):
        features = []
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            features.append(x)
        
        # Apply FPN to the encoder features
        x = None
        for fpn_block, feature in zip(self.fpn_encoder, reversed(features)):
            x = fpn_block(feature, x)
        
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 128, 10, 20)  # Initial shape for the decoder
        
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)
        
        x = self.final_layer(x)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

#class Network(nn.Module):
#    def __init__(self, name):
#        super(Network, self).__init__()
#        self.name = name
#
#    def init_weights(self, m):
#        if isinstance(m, nn.Linear):
#            torch.nn.init.orthogonal_(m.weight.data)
#            if m.bias is not None:
#                torch.nn.init.constant_(m.bias.data, 0)
#        elif isinstance(m, nn.Conv2d):
#            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#            if m.bias is not None:
#                nn.init.constant_(m.bias, 0)

# 
class Conv1dVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(Conv1dVAE, self).__init__()
        # 
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
            # MaxPool1d
        )
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(256 * 45, latent_dim)
        self.fc_logvar = nn.Linear(256 * 45, latent_dim)

        # 
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
            nn.Sigmoid()  #
        )

    def encode(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # logvar
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

# Modify Actor class
class Actor(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Actor, self).__init__(name)
        # --- define layers here ---

        # Load the first VAE model (for image1)
        self.vae1 = Conv2dVAE(latent_dim=128)
        self.vae1.load_state_dict(torch.load('vae_model1.pth', map_location=torch.device('cpu')), strict=False)
        #self.vae1.eval()
        self.vae1.train()

        # Load the second VAE model (for image2)
        self.vae2 = Conv2dVAE(latent_dim=128)
        self.vae2.load_state_dict(torch.load('vae_model2.pth', map_location=torch.device('cpu')), strict=False)
        #self.vae2.eval()
        self.vae2.train()

        self.vae3 = Conv1dVAE(latent_dim=32)
        self.vae3.load_state_dict(torch.load('vae_model3.pth', map_location=torch.device('cpu')), strict=False)
        self.vae3.train()        

        # Fix encoder parameters
        #for param in self.vae1.parameters():
        #    param.requires_grad = False
        #for param in self.vae2.parameters():
        #    param.requires_grad = False

        # Define fully connected layers to pass through after combining additional state information
        conv_output_size = 128
        lidar_output_size = 32
        
        self.fc1 = nn.Linear(conv_output_size * 2 + lidar_output_size + 4, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln2 = nn.LayerNorm(hidden_size // 2)
        
        self.fc3 = nn.Linear(hidden_size // 2, action_size)
        
        #self.se = nn.Sequential(
        #    nn.Linear(hidden_size, hidden_size // 16),
        #    nn.ReLU(),
        #    nn.Linear(hidden_size // 16, hidden_size),
        #    nn.Sigmoid()
        #)
        
        self.dropout = nn.Dropout(p=0.1)
        
        # VAE
        self.init_weights(self.fc1)
        self.init_weights(self.fc2)
        self.init_weights(self.fc3)
        self.init_weights(self.ln1)
        self.init_weights(self.ln2)

    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    #def init_weights(self, m):
    #    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
    #        nn.init.xavier_uniform_(m.weight)
    #        if m.bias is not None:
    #            nn.init.zeros_(m.bias)

    def forward(self, states, visualize=False):
        depth_image_size = 80 * 160
        lidar_output_size = 360

        if states.dim() == 2:
            # When batch is present
            li1 = states[:, :depth_image_size]
            li2 = states[:, depth_image_size:2 * depth_image_size]
            li3 = states[:, 2 * depth_image_size:2 * depth_image_size + lidar_output_size]
            s1 = states[:, -4:]
        else:
            # When batch is not present
            li1 = states[:depth_image_size].view(1, -1)
            li2 = states[depth_image_size:2 * depth_image_size].view(1, -1)
            li3 = states[2 * depth_image_size:2 * depth_image_size + lidar_output_size].view(1, -1)
            s1 = states[-4:].view(1, -1)

        #print("li1", li1, li1.min(), li1.max(), li1.shape, li1.dtype)
        #print("li3", li3, li3.min(), li3.max(), li3.shape, li3.dtype)

        darkness_factor = 0.2  # For 10% brightness


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
        li3_reshaped = li3.view(li3.size(0), 1, -1)

        # Apply darkness_factor
        li1_imaged = li1_imaged * darkness_factor
        li2_imaged = li2_imaged * darkness_factor        


        # Extract latent vector through the first VAE encoder
        with torch.enable_grad():
            mu1, logvar1 = self.vae1.encode(li1_imaged)
            z1 = self.vae1.reparameterize(mu1, logvar1)

            mu2, logvar2 = self.vae2.encode(li2_imaged)
            z2 = self.vae2.reparameterize(mu2, logvar2)

            mu3, logvar3 = self.vae3.encode(li3_reshaped)
            z3 = self.vae3.reparameterize(mu3, logvar3)

        # Combine the two latent vectors
        #latent_vector = torch.cat((mu1, mu2), dim=1)  # (batch_size, 96)

        # Combine additional state information
        #li_s1 = torch.cat((latent_vector, s1), dim=1)  # (batch_size, 100)

        # Pass through fully connected layers
        x = torch.cat((z1, z2, z3, s1), dim=1)
        
        # Decision layers with residual connection
        #identity = x
        x = F.gelu(self.ln1(self.fc1(x)))
        #x = x * self.se(x)
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

# Modify Critic class
class Critic(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Critic, self).__init__(name)
        # --- define layers here ---

        # --- Integrate VAE encoder ---
        # Load the first VAE model (for image1)
        self.vae1 = Conv2dVAE(latent_dim=128)
        self.vae1.load_state_dict(torch.load('vae_model1.pth', map_location=torch.device('cpu')), strict=False)
        #self.vae1.eval()
        self.vae1.train()

        # Load the second VAE model (for image2)
        self.vae2 = Conv2dVAE(latent_dim=128)
        self.vae2.load_state_dict(torch.load('vae_model2.pth', map_location=torch.device('cpu')), strict=False)
        #self.vae2.eval()
        self.vae2.train()

        self.vae3 = Conv1dVAE(latent_dim=32)
        self.vae3.load_state_dict(torch.load('vae_model3.pth', map_location=torch.device('cpu')), strict=False)
        self.vae3.train()

        # Fix encoder parameters to prevent training
        #for param in self.vae1.parameters():
        #    param.requires_grad = False
        #for param in self.vae2.parameters():
        #    param.requires_grad = False

        # --- Define Critic network layers ---
        conv_output_size = 128
        lidar_output_size = 32
        
        self.fc1 = nn.Linear(conv_output_size * 2 + lidar_output_size + 4, 256)
        self.ln1 = nn.LayerNorm(256)
        
        self.fc2 = nn.Linear(action_size, 256)
        self.ln2 = nn.LayerNorm(256)
        
        self.fc3 = nn.Linear(512, 512)
        self.ln3 = nn.LayerNorm(512)
        
        self.fc4 = nn.Linear(512, 1)
        
        #self.se = nn.Sequential(
        #    nn.Linear(256, 256 // 16),
        #    nn.ReLU(),
        #    nn.Linear(256 // 16, 256),
        #    nn.Sigmoid()
        #)
        
        self.dropout = nn.Dropout(p=0.2)
        
        # VAE
        self.init_weights(self.fc1)
        self.init_weights(self.fc2)
        self.init_weights(self.fc3)
        self.init_weights(self.fc4)
        self.init_weights(self.ln1)
        self.init_weights(self.ln2)
        self.init_weights(self.ln3)

    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, states, actions, visualize=False):
        depth_image_size = 80 * 160
        lidar_output_size = 360

        if states.dim() == 2:
            # When batch is present
            li1 = states[:, :depth_image_size]
            li2 = states[:, depth_image_size:2 * depth_image_size]
            li3 = states[:, 2 * depth_image_size:2 * depth_image_size + lidar_output_size]
            s1 = states[:, -4:]
        else:
            # When batch is not present
            li1 = states[:depth_image_size].view(1, -1)
            li2 = states[depth_image_size:2 * depth_image_size].view(1, -1)
            li3 = states[2 * depth_image_size:2 * depth_image_size + lidar_output_size].view(1, -1)
            s1 = states[-4:].view(1, -1)

        # Normalize data in the same way as during VAE training
        #li1_normalized = li1 / 255.0  # Scale pixel values to [0,1]

        # Convert to channel and image dimensions
        li1_imaged = li1.view(li1.size(0), 1, 80, 160)
        li2_imaged = li2.view(li2.size(0), 1, 80, 160)
        li3_reshaped = li3.view(li3.size(0), 1, -1)


        darkness_factor = 0.2  # For 10% brightness
 
        # Apply darkness_factor
        li1_imaged = li1_imaged * darkness_factor
        li2_imaged = li2_imaged * darkness_factor         



        # Extract latent vector through the first VAE encoder
        with torch.enable_grad():
            mu1, logvar1 = self.vae1.encode(li1_imaged)
            z1 = self.vae1.reparameterize(mu1, logvar1)

            mu2, logvar2 = self.vae2.encode(li2_imaged)
            z2 = self.vae2.reparameterize(mu2, logvar2)

            mu3, logvar3 = self.vae3.encode(li3_reshaped)
            z3 = self.vae3.reparameterize(mu3, logvar3)

        # Combine the two latent vectors
        #latent_vector = torch.cat((mu1, mu2), dim=1)  # (batch_size, 96)

        # Combine additional state information
        #li_s1 = torch.cat((latent_vector, s1), dim=1)  # (batch_size, 100)

        # Pass through Critic network layers
        xs = torch.cat((z1, z2, z3, s1), dim=1)
        
        xs = self.fc1(xs)
        xs = self.ln1(xs)
        xs = F.gelu(xs)
        #xs = xs * self.se(xs)
        xs = self.dropout(xs)
        
        xa = self.fc2(actions)
        xa = self.ln2(xa)
        xa = F.gelu(xa)
        xa = self.dropout(xa)
        
        x = torch.cat((xs, xa), dim=1)
        
        x = self.fc3(x)
        x = self.ln3(x)
        x = F.gelu(x)
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