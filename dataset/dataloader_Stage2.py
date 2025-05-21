import glob
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from natsort import natsorted
import cv2
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F

# PATH
TRAIN_IMG = r'data/train/images'
TRAIN_MASK = r'data/train/masks'

VAL_IMG = r'data/val/images'
VAL_MASK = r'data/val/masks'

imagenet_std    = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1)
imagenet_mean   = torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1)

batch_size = 4
batchsize_val = 4
resize_w = 256
resize_h = 256


class Dataset(Dataset):
    def __init__(self, mode="train", use_normalization=True):

        self.mode = mode
        self.use_normalization = use_normalization

        if mode == 'train':
            self.files_wholeimg = natsorted(sorted(glob.glob(TRAIN_IMG + "/*")))
            self.files_mask = natsorted(sorted(glob.glob(TRAIN_MASK + "/*")))
        else:
            self.files_wholeimg = natsorted(sorted(glob.glob(VAL_IMG + "/*")))
            self.files_mask = natsorted(sorted(glob.glob(VAL_MASK + "/*")))

        transform_list = [
            T.ToPILImage(),
            T.Resize((resize_w, resize_h)),
            T.ToTensor()
        ]

        if self.use_normalization:
            transform_list.append(
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )

        self.transform = T.Compose(transform_list)

    def __getitem__(self, index):
        try:
            wholeimage = cv2.imread(self.files_wholeimg[index])[..., ::-1]
            mask = cv2.imread(self.files_mask[index], cv2.IMREAD_GRAYSCALE)

            # 应用转换
            wholeimage = self.transform(wholeimage)
            mask = cv2.resize(mask, (resize_w, resize_h))
            mask = torch.from_numpy(mask).unsqueeze(0).float() / 255

            mask = mask.round()

            return wholeimage.float(), mask.float()

        except Exception as e:
            print(f"Error loading data at index {index}: {e}")
            return self.__getitem__(index + 1) if index + 1 < len(self.files_wholeimg) else None

    def __len__(self):
        return len(self.files_wholeimg)


# Training data loader
trainloader = DataLoader(
    Dataset(mode="train"),
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=2,
    drop_last=True
)


valloader = DataLoader(
    Dataset(mode="val"),
    batch_size=batchsize_val,
    shuffle=False,
    pin_memory=True,
    num_workers=1,
    drop_last=True
)