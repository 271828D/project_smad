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
parser.add_argument('--augmentation', type=str, default='soft', help="Data augmentatios for the training process. Options soft, medium and hard")

params = parser.parse_args()

data_dir = params.data_dir
num_epochs = params.num_epochs
batch_size = params.batch_size
img_size = params.img_size
class_num = params.class_num
lr = params.lr
model = params.model
augmentation = params.augmentation
#########################################################
if th.cuda.is_available():
  device = "cuda:0"
else:
  device = "cpu"
#########################################################
# TO DO: 
# agregar wandb monitor

#########################################################
# TO DO: 
# agregar tres opciones de augmentations: soft, medium, hard
train_path = os.path.join(data_dir, "train")
val_path = os.path.join(data_dir, "validation")
test_path = os.path.join(data_dir, "test")

if augmentation == "soft":
   transf = v2.Compose([
                        v2.Resize(size=(img_size,img_size)),
                        v2.RandomHorizontalFlip(p=.5),
                        v2.ToTensor(),
                        # transforms.PILToTensor()
                        ])
else:
   print("Not yet developed medium and hard")
   transf = v2.Compose([
                        v2.Resize(size=(img_size,img_size)),
                        v2.RandomHorizontalFlip(p=.5),
                        v2.ToTensor(),
                        # transforms.PILToTensor()
                        ])

train_data = ImageFolder(root=data_dir, transform=transf, 
                #    is_valid_file = checkImage
                   )
generator = th.Generator().manual_seed(42)
train_data, val_data = th.utils.data.random_split(dataset=train_data, lengths=[.8,.2], generator=generator)
print('train:',len(train_data))
print('val:',len(val_data))
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
#########################################################
############ Models ##################
if model == "efficientnetb3":
    ######## EfficientNet B3 #############
    model = tv.models.efficientnet_b3(weights='IMAGENET1K_V1')
    print('Original model classifier:', model.classifier)
    model.classifier = nn.Sequential(
                                                nn.Dropout(p=.2, inplace=True),
                                                nn.Linear(in_features=1536, out_features=class_num, bias=True),
                                                nn.Softmax(dim=1)
                                                )
    print('Modified models classifier:', model.classifier)
    model.to(device)
elif model == 'mobilenetv3':
    ######## MobileNet V3 #############
    model = tv.models.mobilenet_v3_large(weights='IMAGENET1K_V1')
    print('Original model classifier:', model.classifier)
    model.classifier = nn.Sequential(
                                                    nn.Linear(in_features=960, out_features=1280, bias=True),
                                                    nn.Hardswish(inplace=True),
                                                    nn.Dropout(p=.2, inplace=True),
                                                    nn.Linear(in_features=1280, out_features=class_num, bias=True),
                                                    nn.Softmax(dim=1)
                                                )
    print('Modified models classifier:', model.classifier)
elif model == 'swin_transformer':
    ######## Swin Transformer #############
    model = tv.models.swin_t(weights='IMAGENET1K_V1')
    print('Head classifier:', model.head)
    model.head = nn.Sequential(
                                nn.Linear(in_features=768, out_features=class_num, bias=True),
                                nn.Softmax(dim=1)
                                            )
    print('Modified head classifier:', model.head)
#######################################
# Criterion
loss_fn = nn.CrossEntropyLoss()

# Optimizer
optimizer = th.optim.Adam(lr=lr, params=model.parameters(), weight_decay=1e-4)

# Metric
metric_precision = MulticlassPrecision(num_classes=class_num)
metric_recall = MulticlassRecall(num_classes=class_num)
#########################################
print("[INFO] training the network...")
startTime = time.time()
min_valid_loss = np.inf
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="project s-mad",
    # track hyperparameters and run metadata
    config={
            "model": model,
            "batch_size": batch_size,
            "learning_rate": lr,
            "epochs": epochs,
            "start_time": startTime,
    }
)

for epoch in tqdm(range(num_epochs), desc="Epoch", position=1):
   train_losses = np.array([])
   train_precision = []
   train_recall = []

   for batch, labels in (pbar:= tqdm(train_loader, desc="Batch", position=0)):
      batch = batch.to(device)
      labels = labels.to(device)
      # Setting grad in zero
      optimizer.zero_grad(set_to_none=True) # optimizer.zero_grad()
      # Model estimation
      transformed = model.forward(batch).to(device)
    #   transformed = th.argmax(transformed, dim=1)
      # Criterion
      loss = loss_fn(transformed, labels).to(device)
      # Metrics
      precision_ = metric_precision(th.argmax(transformed, dim=1), labels)
      recall_ = metric_recall(th.argmax(transformed, dim=1), labels)
      # Backpropagation
      loss.backward()
      optimizer.step()
    #   train_losses.append(loss)
      train_losses = np.append(train_losses, loss.detach().numpy())
    #   TO DO: add append of the metrics calculation
      # Model evaluation
      model.eval()
      with th.no_grad():
         val_losses = np.array([])
         val_precision = []
         val_recall = []
         for batch, labels in val_loader:
            batch = batch.to(device)
            labels = labels.to(device)
            transformed = model.forward(batch).to(device)
            val_loss = loss_fn(transformed, labels).to(device) # change: labels:long to labels.type('torch.FloatTensor').to(device)
            val_precision_ = metric_precision(transformed, labels)
            val_recall_ = metric_recall(transformed, labels)
            # print('val-loss:', loss.item())
            # val_losses.append(loss)
            val_losses = np.append(val_losses, val_loss.detach().numpy())
            # val_precision.append(val_precision_)
            # val_recall.append(val_recall_)
    #   Print batch information
      print(f'\t Epoch {num_epochs+1} \t loss: {np.sum(train_losses) / len(train_losses)} \t val-loss: {np.sum(val_losses) / len(val_losses)}')
    #   wandb tracking
      wandb.log({
    "loss": np.mean(train_losses),
    "val_loss": np.mean(val_losses),
    })
    #   Save the best model
      if min_valid_loss > np.sum(val_losses) / len(val_losses):
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{np.sum(val_losses) / len(val_losses):.6f}) \t Saving The Model')
        min_valid_loss = np.sum(val_losses) / len(val_losses)
        th.save(model.state_dict(), model+'_'+str(num_epochs))
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))   

# print('dataset path:', data_dir, '\r')
# print('num-epochs:', num_epochs, '\r')
# print('batch_size:', batch_size, '\r')
# print('img_size:', img_size, '\r')
# print('class num:', class_num, '\r')
# print('lr:', lr, '\r')
# print('model:', model, '\r')
# print('augmentation:', augmentation, '\r')