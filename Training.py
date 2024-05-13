import torch as th
import torch.nn as nn
import torchvision as tv
from torchmetrics.classification import BinaryAUROC, BinaryPrecision, BinaryRecall
from tqdm.auto import tqdm
import numpy as np

import time
import wandb
import random
import os
import argparse

from Dataloader.Dataloader import train_loader, val_loader

#######################
# Hyper-Parameters
lr = 3e-4
batch_size = 32
optimizer = "adam"
epochs=2

# gpu or cpu training
if th.cuda.is_available():
  device = "cuda:0"
  # device = th.cuda.get_device_name()
else:
  device = "cpu"

############ Models ##################
######## EfficientNet B3 #############
efficientnetb3 = tv.models.efficientnet_b3(weights='IMAGENET1K_V1')
print('Original model classifier:', efficientnetb3.classifier)
efficientnetb3.classifier = nn.Sequential(
                                              nn.Dropout(p=.2, inplace=True),
                                              nn.Linear(in_features=1536, out_features=6, bias=True),
                                              nn.Softmax(dim=1)
                                            )
print('Modified models classifier:', efficientnetb3.classifier)
efficientnetb3.to(device)

######## MobileNet V3 #############
# mobilenetv3_large = tv.models.mobilenet_v3_large(weights='IMAGENET1K_V1')
# print('Original model classifier:', mobilenetv3_large.classifier)
# mobilenetv3_large.classifier = nn.Sequential(
#                                                 nn.Linear(in_features=960, out_features=1280, bias=True),
#                                                 nn.Hardswish(inplace=True),
#                                                 nn.Dropout(p=.2, inplace=True),
#                                                 nn.Linear(in_features=1280, out_features=6, bias=True),
#                                                 nn.Softmax(dim=1)
#                                             )
# print('Modified models classifier:', mobilenetv3_large.classifier)

######## Swin Transformer #############
# model = tv.models.swin_t(weights='IMAGENET1K_V1')
# print('Head classifier:', model.head)
# model.head = nn.Sequential(
#                             nn.Linear(in_features=768, out_features=6, bias=True),
#                             nn.Softmax(dim=1)
#                                         )
# print('Modified head classifier:', model.head)
#######################################

# Criterion
loss_fn = nn.CrossEntropyLoss()

# Optimizer
optimizer = th.optim.Adam(lr=lr, params=efficientnetb3.parameters(), weight_decay=1e-4)

# Metric
metric_auroc = BinaryAUROC(thresholds=None)
metric_precision = BinaryPrecision(threshold=.5)
metric_recall = BinaryRecall(threshold=.5)

##### Hyperparameter search #####
#################################
parser = argparse.ArgumentParser(
                    prog='Traning CNNs',
                    description='Train the CNNs and Hyperparameter searching',
                    epilog='Models used efficientnetb3 and mobilenetv3')

parser.add_argument('--learning_rate', type=float, default=lr)
# parser.add_argument('--optimizer', type=str, default=optimizer)
parser.add_argument('--batch_size', type=int, default=batch_size)
# parser.add_argument('--model', type=str, default=model)
# parser.add_argument('--cc_loss_weight', type=float, default=cc_loss_weight)


args = parser.parse_args()

lr = args.learning_rate
# optimizer = args.optimizer
batch_size = args.batch_size
# model = args.model
# cc_loss_weight = args.cc_loss_weight

#################################

#######################################
print("[INFO] training the network...")
startTime = time.time()

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="project s-mad",

    # track hyperparameters and run metadata
    config={
          "learning_rate": lr,
          "epochs": epochs,
          "start_time": startTime,
    }
)

# best_model_state = None
# best_metric_value = float('inf')

for epoch in tqdm(range(epochs), desc="Epoch", position=1):
  
  # get a new batch
  # epoch_images = []
  train_losses = np.array([])
  # train_auroc = np.array([])
  train_precision = np.array([])
  train_recall = np.array([])

  for batch, labels in (pbar:= tqdm(train_loader, desc="batch", position=0)):
    batch = batch.to(device)
    labels = labels.to(device)

    optimizer.zero_grad(set_to_none=True) # optimizer.zero_grad()
    transformed = efficientnetb3.forward(batch).to(device)

    # Loss calculation
    loss = loss_fn(transformed, labels).to(device)

    # Metrics calculation
    # auroc_ = metric_auroc(transformed, labels)
    precision_ = metric_precision(transformed, labels)
    recall_ = metric_recall(transformed, labels)
    
    # Backpropagation step
    loss.backward()
    optimizer.step()

    # train_losses.append(loss)
    train_losses = np.append(train_losses, loss.detach().numpy())
    # train_auroc = np.append(train_auroc, auroc_.detach().numpy())
    train_precision = np.append(train_precision, precision_.detach().numpy())
    train_recall = np.append(train_recall, recall_.detach().numpy())

    # print(f"Epoch {1+epoch:5} - Loss: {loss.item():<7.5}")  # print(f"Epoch {1+epoch:5} - Loss: {loss.item():<7.5}")
    
    efficientnetb3.eval()
    with th.no_grad():
       val_losses = np.array([])
      #  val_auroc = np.array([])
       val_precision = np.array([])
       val_recall = np.array([])
      #  result_images = []
       
       for batch, labels in val_loader:
        transformed = efficientnetb3.forward(batch).to(device)
        val_loss = loss_fn(transformed, labels).to(device) # change: labels:long to labels.type('torch.FloatTensor').to(device)
        # val_auroc_ = metric_auroc(transformed, labels)
        val_precision_ = metric_precision(transformed, labels)
        val_recall_ = metric_recall(transformed, labels)
        #  val_losses.append(val_loss.item())

        val_losses = np.append(val_losses, val_loss.detach().numpy())
        val_auroc = np.append(val_auroc, val_auroc_.detach().numpy())
        val_precision = np.append(val_precision, val_precision_.detach().numpy())
        val_recall = np.append(val_recall, val_recall_.detach().numpy())

        # print(f"Epoch {1+epoch:5} - Loss: {val_loss.item():<7.5}")

        # Metric calculation
        # ADD METRIC

  
  wandb.log({
    "loss": np.mean(train_losses),
    # "auroc": np.mean(train_auroc),
    # 'precision': np.mean(train_precision),
    # 'recall': np.mean(train_recall),
    # "lr": lr,
    "val_loss": np.mean(val_losses),
    # "val_Acc": np.mean(test_accs),
    })

endTime = time.time()
endTime = endTime
running_time = endTime - startTime
running_time = running_time
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))


# wandb.log({
#     "end_time": endTime,
#     "running_time": running_time
#     })



