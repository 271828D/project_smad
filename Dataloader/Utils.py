import os
from PIL import Image
import torch as th
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode

class FeretDataset(th.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = os.listdir(data_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.images[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image
    
    def load_image(self, idx):
        img_name = os.path.join(self.data_dir, self.images[idx])
        """Reading img"""
        print(self.imgs_dir, self.imgs[idx])
        # img_path = os.path.join(self.imgs_dir, image_path)
        image = read_image(img_name)
        image = image/255.0
        image = image.to(th.float32)
        image = image.permute(2, 0 ,1)
        # image = np.transpose(image, (2,0,1))
        # image = image.astype(np.float32)
        # image = th.from_numpy(image)
        return image