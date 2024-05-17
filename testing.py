import argparse
import torch as th
import torch.nn as nn
import torchvision as tv
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassConfusionMatrix
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import numpy as np

import time
import wandb
import random
import os
import argparse

# from Dataloader.Dataloader import train_loader, val_loader

model = 'asdf'
lr = 3e-2
epochs = 1

parser = argparse.ArgumentParser()

parser.add_argument('--data-dir', type=str, default=None, help='path of /dataset/')
parser.add_argument('--num-epochs', type=int, default=1)
parser.add_argument('--batch-size', type=int, default=32, help='total batch size for all GPUs')
parser.add_argument('--img-size', type=int, default=[224, 224], help='img sizes')
parser.add_argument('--class-num', type=int, default=4, help='class num')
parser.add_argument("--lr", type=float, default= .0003, help="learning rate")
parser.add_argument("--model", type=str, default="efficientnet-b3", help="model type available efficientnetb3, mobilenetv3, swin_transformer")
parser.add_argument('--weights-path', type=str, default=None, help="Add the path to the weights folder")

params = parser.parse_args()

data_dir = params.data_dir
num_epochs = params.num_epochs
batch_size = params.batch_size
img_size = params.img_size
class_num = params.class_num
lr = params.lr
model = params.model
weights_path = params.weights_path

print('dataset path:', data_dir, '\r')
print('num-epochs:', num_epochs, '\r')
print('batch_size:', batch_size, '\r')
print('img_size:', img_size, '\r')
print('class num:', class_num, '\r')
print('lr:', lr, '\r')
print('model:', model, '\r')
print('weights_path:', weights_path, '\r')
#########################################################
if th.cuda.is_available():
  device = "cuda:0"
else:
  device = "cpu"
#########################################################
test_path = os.path.join(data_dir, "test")
transf = v2.Compose([
                        v2.Resize(size=(img_size,img_size)),
                        v2.RandomHorizontalFlip(p=.5),
                        v2.ToTensor(),
                        # transforms.PILToTensor()
                        ])
try: 
   if os.path.exists(test_path):
    test_data = ImageFolder(root=test_path, transform=transf, 
                        #    is_valid_file = checkImage
                        )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
except:
   raise Exception("The test path doesn't exist")
############ Models ##################
if model == "efficientnetb3":
    model_name = model
    ######## EfficientNet B3 #############
    model = tv.models.efficientnet_b3(weights='IMAGENET1K_V1')
    model.classifier = nn.Sequential(
                                                nn.Dropout(p=.2, inplace=True),
                                                nn.Linear(in_features=1536, out_features=class_num, bias=True),
                                                nn.Softmax(dim=1)
                                                )
    model.to(device)
elif model == 'mobilenetv3':
    model_name = model
    ######## MobileNet V3 #############
    model = tv.models.mobilenet_v3_large(weights='IMAGENET1K_V1')
    model.classifier = nn.Sequential(
                                                    nn.Linear(in_features=960, out_features=1280, bias=True),
                                                    nn.Hardswish(inplace=True),
                                                    nn.Dropout(p=.2, inplace=True),
                                                    nn.Linear(in_features=1280, out_features=class_num, bias=True),
                                                    nn.Softmax(dim=1)
                                                )
    model.to(device)
elif model == 'swin_transformer':
    model_name = model
    ######## Swin Transformer #############
    model = tv.models.swin_t(weights='IMAGENET1K_V1')
    print('Head classifier:', model.head)
    model.head = nn.Sequential(
                                nn.Linear(in_features=768, out_features=class_num, bias=True),
                                nn.Softmax(dim=1)
                                            )
    model.to(device)
model.load_state_dict(th.load(weights_path))
#######################################
# Criterion
loss_fn = nn.CrossEntropyLoss().to(device)
#######################################
print("[INFO] training the network...")
startTime = time.time()

model.eval()
test_losses = np.array([])
for batch, labels in (pbar:= tqdm(test_data, desc="Batch", position=0)):
   with th.no_grad():
        batch = batch.to(device)
        labels = labels.to(device)
        transformed = model.forward(batch)
        loss = loss_fn(transformed, labels).to(device)
        train_losses = np.append(train_losses, loss.detach().cpu())
print(f'\t test loss: {np.sum(test_losses) / len(test_losses)}')
      