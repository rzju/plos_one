import os
import numpy as np
from PIL import Image
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from model.unet import UNet
from dataset.OIMHS import OIMHSDataset
from utils import dice_coefficient_multiclass, jaccard_index
from loss.GDiceLoss import GDiceLoss
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from model.unet import UNet
from model.vmamba import VSSM
from model.vmunet import VMUNet
from torch.optim.lr_scheduler import CosineAnnealingLR



images_output_dir = './output/images'
masks_output_dir = './output/masks'

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# 加载数据集
dataset = OIMHSDataset(images_output_dir, transform=transform)

# 定义数据加载器
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

# 打印分配给训练和验证的数据集大小
print(f"Training data size: {len(train_indices)}, Validation data size: {len(val_indices)}")

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

# 打印验证数据加载器的长度以确认不为空
print(f"Validation loader length: {len(validation_loader)}")
print(f"train_loader loader length: {len(train_loader)}")

mamba_params = {
    'in_chans': 3, 
    'num_classes': 4, 
    'depths': [ 2, 2, 15, 2 ], 
    'dims': [96, 192, 384, 768],
}
model_cfg = {
    'num_classes': 4, 
    'input_channels': 3, 
    # ----- VM-UNet ----- #
    'depths': [2,2,2,2],
    'depths_decoder': [2,2,2,1],
    'drop_path_rate': 0.2,
    'load_ckpt_path': './checkpoints/vmamba_small_e238_ema.pth',
}
model = VMUNet(
num_classes=model_cfg['num_classes'],
input_channels=model_cfg['input_channels'],
depths=model_cfg['depths'],
depths_decoder=model_cfg['depths_decoder'],
drop_path_rate=model_cfg['drop_path_rate'],
load_ckpt_path=model_cfg['load_ckpt_path'],
)
model.load_from()
# print('load pre-trained model weight success')
# 初始化模型、损失函数和优化器
# model = VSSM(**mamba_params)
# model = UNet(n_channels=3, n_classes=4)


# 预训练模型
# checkpoint = 'https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth'
# load_dict = torch.hub.load_state_dict_from_url(checkpoint, progress=True)
# checkpoint_path = './checkpoints/vssm_base_0229_ckpt_epoch_237.pth'
# load_dict = torch.load(checkpoint_path)['model']
# model_dict = model.state_dict()

# for key1, key2 in zip(load_dict.keys(), model_dict.keys()):
#     print(key1, key2)

# for key, value in load_dict.items():
#     if 'outc' not in key: 
#         model_dict[key] = value

# model.load_state_dict(model_dict)




ce_loss = nn.CrossEntropyLoss()
gdice_loss = GDiceLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练和验证循环
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)

num_epochs = 200
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# 设置学习率调度器
batches_per_epoch = dataset_size // batch_size
T_max = num_epochs * batches_per_epoch
scheduler = CosineAnnealingLR(optimizer, T_max, eta_min=1e-5) 

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_dice = np.zeros(4)
    running_jaccard = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)


        optimizer.zero_grad()
        outputs = model(images)

        # loss = gdice_loss(outputs, masks)

        # 确保输出和掩码形状一致
        if outputs.shape[-2:] != masks.shape[-2:]:
            masks = nn.functional.interpolate(masks, size=outputs.shape[-2:], mode="nearest")

        outputs = outputs.permute(0, 2, 3, 1).contiguous().view(-1, outputs.size(1))
        masks = masks.permute(0, 2, 3, 1).contiguous().view(-1, masks.size(1))

        if masks.size(1) > 1:
            masks = masks.argmax(dim=1)

        loss = ce_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        dice = dice_coefficient_multiclass(preds, masks, 4)
        running_dice += dice

    epoch_loss = running_loss / len(train_loader)
    epoch_dice = running_dice / len(train_loader)
    # epoch_jaccard = running_jaccard / len(train_loader)

    model.eval()
    val_loss = 0.0
    val_dice = np.zeros(4)
    val_jaccard = 0.0
    with torch.no_grad():
        for images, masks in validation_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)

            if outputs.shape[-2:] != masks.shape[-2:]:
                masks = nn.functional.interpolate(masks, size=outputs.shape[-2:], mode="nearest")

            outputs = outputs.permute(0, 2, 3, 1).contiguous().view(-1, outputs.size(1))
            masks = masks.permute(0, 2, 3, 1).contiguous().view(-1, masks.size(1))

            if masks.size(1) > 1:
                masks = masks.argmax(dim=1)

            loss = ce_loss(outputs, masks)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            dice = dice_coefficient_multiclass(preds, masks, 4)
            val_dice += dice
            # val_jaccard += jaccard

    if len(validation_loader) > 0:
        val_loss /= len(validation_loader)
        val_dice /= len(validation_loader)
        # val_jaccard /= len(validation_loader)
    else:
        val_loss, val_dice, val_jaccard = float('nan'), float('nan'), float('nan')

    # print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, Training Dice: {epoch_dice:.4f}, Validation Dice: {val_dice:.4f}, Training Jaccard: {epoch_jaccard:.4f}, Validation Jaccard: {val_jaccard:.4f}')
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, Training Dice: {epoch_dice}, Validation Dice: {val_dice}')

    scheduler.step()

    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1:02d}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        
final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
torch.save(model.state_dict(), final_model_path)
print("Final model saved.")

