import os
from PIL import Image
import torch as th
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.io import read_image, ImageReadMode
from torchvision.datasets import ImageFolder

# dataset_path = 'D:\\s-mad-dataset\\FERET\\npp\\dataset_test'
dataset_path = 'C:\\Users\\BenaventeD\\data\\s-mad-dataset\\preprocessed-frgc\\dataset_test\\'

transf = v2.Compose([
                        v2.Resize(size=(224,224)),
                        v2.RandomHorizontalFlip(p=.5),
                        v2.ToTensor(),
                        # transforms.PILToTensor()
                            ])

train_data = ImageFolder(root = dataset_path, transform = transf, 
                #    is_valid_file = checkImage
                   )

generator = th.Generator().manual_seed(42)
train_data, val_data = th.utils.data.random_split(dataset=train_data, lengths=[.8,.2], generator=generator)
val_data, test_data = th.utils.data.random_split(dataset=val_data, lengths=[.5,.5], generator=generator)

print(type(train_data))
print('train:',len(train_data))
print('val:',len(val_data))
print('test:',len(test_data))

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

x,y = next(iter(train_loader))
print('train dim:', x.size(), y.size())

x,y = next(iter(val_loader))
print('val dim:', x.size(), y.size())

x,y = next(iter(test_loader))
print('test dim:', x.size(), y.size())