from torch.utils.data import Dataset, DataLoader
from glob import glob
import os
from PIL import Image
import torch
import torch.nn as nn

class OIMHSDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_files = sorted(glob(os.path.join(image_dir, '*.png')))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_files = self.image_files[idx]
        mask_files = image_files.replace('images', 'masks').replace('oct', 'mask')

        image = Image.open(image_files).convert('RGB')
        mask = Image.open(mask_files).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            mask = (mask * 255).long()

            mask = torch.clamp(mask, min=0, max=3)
            mask = nn.functional.one_hot(mask.squeeze(0), num_classes=4).permute(2, 0, 1).float()

        return image, mask
