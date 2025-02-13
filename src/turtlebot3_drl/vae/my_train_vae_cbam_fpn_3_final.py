import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

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

# 커스텀 데이터셋 클래스 정의
class PairedDepthDataset(Dataset):
    def __init__(self, input_dirs, target_dir):
        """
        input_dirs: 입력 이미지 디렉토리의 리스트
        target_dir: 타겟 이미지 디렉토리
        """
        self.input_dirs = input_dirs
        self.target_dir = target_dir
        self.file_pairs = []

        for input_dir in input_dirs:
            input_files = sorted([f for f in os.listdir(input_dir)
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.npy'))])
            for input_file in input_files:
                base_name = os.path.splitext(input_file)[0]
                # 타겟 디렉토리에서 동일한 이름의 파일을 찾습니다
                target_file = None
                for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff'):
                    possible_target_file = os.path.join(target_dir, base_name + ext)
                    if os.path.exists(possible_target_file):
                        target_file = possible_target_file
                        break
                if target_file is not None:
                    self.file_pairs.append((os.path.join(input_dir, input_file), target_file))
                else:
                    # 타겟 파일이 없으면 해당 입력 파일을 건너뜁니다
                    continue

        assert len(self.file_pairs) > 0, "No valid file pairs found."
        print(f"Total file pairs: {len(self.file_pairs)}")

        # 변환 정의 (이미지 크기를 (80, 160)으로 변경)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 그레이스케일로 변환
            transforms.Resize((80, 160)),  # 이미지 크기 조정 (높이: 80, 너비: 160)
            transforms.ToTensor()  # 텐서로 변환 및 [0,1]로 스케일링
        ])

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        input_file, target_file = self.file_pairs[idx]

        # 입력 이미지 로드
        if input_file.lower().endswith('.npy'):
            # .npy 파일 로드
            input_array = np.load(input_file)

            # 배열의 형태를 확인하고 2D로 변환
            if input_array.ndim == 3 and input_array.shape[0] == 1:
                # (1, H, W) 형태라면 채널 차원 제거
                input_array = np.squeeze(input_array, axis=0)
            elif input_array.ndim == 3 and input_array.shape[2] == 1:
                # (H, W, 1) 형태라면 마지막 축 제거
                input_array = np.squeeze(input_array, axis=2)
            elif input_array.ndim == 1:
                # 1D 배열이라면 예상되는 크기로 재구성
                expected_size = 80 * 160
                if input_array.size == expected_size:
                    input_array = input_array.reshape((80, 160))
                else:
                    # 예상치 못한 크기일 경우, 다음 샘플로 넘어갑니다
                    return self.__getitem__((idx + 1) % len(self))
            elif input_array.ndim == 2:
                # 이미 (80, 160) 형태라면 그대로 사용
                pass
            else:
                # 지원되지 않는 배열 형태일 경우, 다음 샘플로 넘어갑니다
                return self.__getitem__((idx + 1) % len(self))

            # 깊이 값 정규화 및 [0,255] 범위로 스케일링
            depth_min = 0.05
            depth_max = 8.0
            input_array = np.clip((input_array - depth_min) / (depth_max - depth_min), 0, 1)
            input_array = (input_array * 255).astype(np.uint8)

            input_image = Image.fromarray(input_array)
        else:
            input_image = Image.open(input_file).convert('L')

        # 변환 적용
        input_image = self.transform(input_image)

        # 타겟 이미지 로드
        if target_file.lower().endswith('.npy'):
            # .npy 파일 로드
            target_array = np.load(target_file)

            # 배열의 형태를 확인하고 2D로 변환
            if target_array.ndim == 3 and target_array.shape[0] == 1:
                # (1, H, W) 형태라면 채널 차원 제거
                target_array = np.squeeze(target_array, axis=0)
            elif target_array.ndim == 3 and target_array.shape[2] == 1:
                # (H, W, 1) 형태라면 마지막 축 제거
                target_array = np.squeeze(target_array, axis=2)
            elif target_array.ndim == 1:
                # 1D 배열이라면 예상되는 크기로 재구성
                expected_size = 80 * 160
                if target_array.size == expected_size:
                    target_array = target_array.reshape((80, 160))
                else:
                    # 예상치 못한 크기일 경우, 다음 샘플로 넘어갑니다
                    return self.__getitem__((idx + 1) % len(self))
            elif target_array.ndim == 2:
                # 이미 (80, 160) 형태라면 그대로 사용
                pass
            else:
                # 지원되지 않는 배열 형태일 경우, 다음 샘플로 넘어갑니다
                return self.__getitem__((idx + 1) % len(self))

            # 깊이 값 정규화 및 [0,255] 범위로 스케일링
            depth_min = 0.05
            depth_max = 8.0
            target_array = np.clip((target_array - depth_min) / (depth_max - depth_min), 0, 1)
            target_array = (target_array * 255).astype(np.uint8)

            target_image = Image.fromarray(target_array)
        else:
            target_image = Image.open(target_file).convert('L')

        # 변환 적용
        target_image = self.transform(target_image)

        return input_image, target_image

# 가중치 초기화 함수
def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# 모델 학습 함수
def train_vae(model_save_path='vae_model2.pth'):
    # 데이터셋 및 데이터로더 생성
    input_dirs = ['images/depth2', 'images/converted_depth2/modified_ori_npy']
    target_dir = 'images/converted_depth2/converted_img'  # 타겟 이미지 디렉토리
    dataset = PairedDepthDataset(input_dirs, target_dir)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)  # 배치 크기를 64로 설정

    # 모델, 옵티마이저, 손실 함수 정의
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 128  # 논문에 따라 128 또는 256으로 설정 가능
    model = Conv2dVAE(latent_dim=latent_dim).to(device)
    model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # 학습률 설정

    num_epochs = 200  # 실제 학습 시 더 많은 에폭 필요
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (input_data, target_data) in enumerate(dataloader):
            input_data = input_data.to(device)
            target_data = target_data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(input_data)

            # 재구성 손실과 KL 발산 계산
            recon_loss = F.mse_loss(recon_batch, target_data, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss

            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        avg_loss = train_loss / len(dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # 모델 가중치 저장
    torch.save(model.state_dict(), model_save_path)
    return model

# 입력 이미지로부터 latent vector를 추출하고, 변형된 이미지를 생성하는 함수
def generate_transformed_images(model, dataset, 
            output_dir='images/generated_transformed_depth2'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for idx in range(len(dataset)):
            input_image, _ = dataset[idx]  # (1, 80, 160)
            input_image = input_image.unsqueeze(0).to(device)  # 배치 차원 추가: (1, 1, 80, 160)
            recon_image, _, _ = model(input_image)
            recon_image = recon_image.squeeze(0).cpu()  # 배치 차원 제거: (1, 80, 160)
            recon_image = recon_image.squeeze(0)  # 채널 차원 제거: (80, 160)
            recon_image = transforms.ToPILImage()(recon_image)
            # 이미지 저장
            input_file, _ = dataset.file_pairs[idx]
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            file_name = base_name + '.png'  # 확장자를 .png로 변경
            output_path = os.path.join(output_dir, file_name)
            recon_image.save(output_path)

if __name__ == '__main__':
    # 모델 학습
    model_save_path = 'vae_model2.pth'  # 저장할 모델 경로
    model = train_vae(model_save_path)

    # 데이터셋 로드
    input_dirs = ['images/depth2', 'images/converted_depth2/modified_ori_npy']
    target_dir = 'images/converted_depth2/converted_img'  # 타겟 이미지 디렉토리
    dataset = PairedDepthDataset(input_dirs, target_dir)

    # 입력 이미지로부터 변형된 이미지 생성 및 저장
    output_dir = 'images/generated_transformed_depth2'
    generate_transformed_images(model, dataset, output_dir)
