import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torchvision.transforms as transforms
import numpy as np
from model.unet import UNet
from model.vmamba import VSSM
from dataset.OIMHS import OIMHSDataset
from utils import dice_coefficient_multiclass

# 参数设置
mamba_params = {
    'in_chans': 3, 
    'num_classes': 4, 
    'depths': [2, 2, 9, 2], 
    'dims': [96, 192, 384, 768],
}

unet_params = {
    'n_channels': 3, 
    'n_classes': 4,
}

# 初始化 Mamba 和 UNet 模型
mamba_model = VSSM(**mamba_params)
unet_model = UNet(**unet_params)

# 数据和数据加载器
images_output_dir = './output/images'
masks_output_dir = './output/masks'

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# 加载数据集
dataset = OIMHSDataset(images_output_dir, transform=transform)

# 数据加载器设置
batch_size = 32
validation_split = 0.3
shuffle_dataset = True
random_seed = 42

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

# 打印数据加载器信息
print(f"Training data size: {len(train_indices)}, Validation data size: {len(val_indices)}")
print(f"Validation loader length: {len(validation_loader)}")
print(f"train_loader loader length: {len(train_loader)}")

# 训练设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mamba_model.to(device)
unet_model.to(device)

ce_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(mamba_model.parameters()) + list(unet_model.parameters()), lr=0.001)

num_epochs = 30
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# 训练和验证循环
for epoch in range(num_epochs):
    mamba_model.train()
    unet_model.train()
    running_loss = 0.0
    running_dice = np.zeros(4)
    
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        
        # 数据首先经过 Mamba 模型
        mamba_outputs = mamba_model(images)
        print(mamba_outputs.shape)
        
        # 然后将 Mamba 的输出作为 UNet 的输入
        unet_outputs = unet_model(mamba_outputs)

        if unet_outputs.shape[-2:] != masks.shape[-2:]:
            masks = nn.functional.interpolate(masks, size=unet_outputs.shape[-2:], mode="nearest")

        unet_outputs = unet_outputs.permute(0, 2, 3, 1).contiguous().view(-1, unet_outputs.size(1))
        masks = masks.permute(0, 2, 3, 1).contiguous().view(-1, masks.size(1))
        
        if masks.size(1) > 1:
            masks = masks.argmax(dim=1)

        loss = ce_loss(unet_outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        preds = torch.argmax(unet_outputs, dim=1)
        dice = dice_coefficient_multiclass(preds, masks, 4)
        running_dice += dice
        
    epoch_loss = running_loss / len(train_loader)
    epoch_dice = running_dice / len(train_loader)
    
    mamba_model.eval()
    unet_model.eval()
    val_loss = 0.0
    val_dice = np.zeros(4)
    
    with torch.no_grad():
        for images, masks in validation_loader:
            images, masks = images.to(device), masks.to(device)
            
            mamba_outputs = mamba_model(images)
            unet_outputs = unet_model(mamba_outputs)
            
            if unet_outputs.shape[-2:] != masks.shape[-2:]:
                masks = nn.functional.interpolate(masks, size=unet_outputs.shape[-2:], mode="nearest")

            unet_outputs = unet_outputs.permute(0, 2, 3, 1).contiguous().view(-1, unet_outputs.size(1))
            masks = masks.permute(0, 2, 3, 1).contiguous().view(-1, masks.size(1))
            
            if masks.size(1) > 1:
                masks = masks.argmax(dim=1)

            loss = ce_loss(unet_outputs, masks)
            val_loss += loss.item()
            
            preds = torch.argmax(unet_outputs, dim=1)
            dice = dice_coefficient_multiclass(preds, masks, 4)
            val_dice += dice

    val_loss /= len(validation_loader)
    val_dice /= len(validation_loader)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, Training Dice: {epoch_dice}, Validation Dice: {val_dice}')

    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1:02d}.pth')
    torch.save({
        'mamba_model_state_dict': mamba_model.state_dict(),
        'unet_model_state_dict': unet_model.state_dict(),
    }, checkpoint_path)
