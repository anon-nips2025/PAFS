import os
import random
from PIL import Image
import torch
from matplotlib import pyplot as plt
from torch.utils import data
from torchvision import transforms


class SimpleDataset(data.Dataset):

    def __init__(self, image_dir, mask_dir, img_size=(256, 256), transform=None):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.transform = transform
        self.image_list = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if
                           fname.endswith(('jpg', 'png'))]
        self.transformMask = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img_path1 = random.choice(self.image_list)
        img_path2 = random.choice(self.image_list)

        mask_path1 = os.path.join(self.mask_dir, os.path.basename(img_path1))

        while img_path1 == img_path2:
            img_path2 = random.choice(self.image_list)

        img1 = Image.open(img_path1).convert('RGB')
        img2 = Image.open(img_path2).convert('RGB')
        mask1 = Image.open(mask_path1).convert('L')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        mask1 = self.transformMask(mask1)
        mask1 = torch.round(mask1)

        return img1, img2, mask1

    def __len__(self):
        return len(self.image_list)


class DataPrefetcher:
    def __init__(self, dataloader):
        self.dataloader = iter(dataloader)
        self.dataset = dataloader.dataset
        self.batch_size = dataloader.batch_size

    def next(self):
        try:
            return next(self.dataloader)
        except StopIteration:
            self.dataloader = iter(torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True))
            return next(self.dataloader)


def create_dataloader(image_dir, mask_dir, batch_size=16, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = SimpleDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader
