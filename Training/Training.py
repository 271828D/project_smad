import torch as th
import torch.nn as nn
import torchvision as tv
from tqdm.auto import tqdm
import time
import wandb
import random
import os
import argparse
from Dataloader.Dataloader import train_loader, val_loader

#######################
# Hyper-Parameters
lr = 3e-4
batch_size = 10
optimizer = "adam"
epochs=100

# Models
mobilenetv3_large = tv.models.mobilenet_v3_large(weights='IMAGENET1K_V1')
print('Original model classifier:', mobilenetv3_large.classifier)
mobilenetv3_large.classifier = nn.Sequential(
                                                nn.Linear(in_features=960, out_features=1280, bias=True),
                                                nn.Hardswish(inplace=True),
                                                nn.Dropout(p=.2, inplace=True),
                                                nn.Linear(in_features=1280, out_features=6, bias=True),
                                                nn.Softmax(dim=1)
                                            )
print('Modified models classifier:', mobilenetv3_large.classifier)

# Criterion
loss_fn = nn.CrossEntropyLoss()

# Optimizer
optimizer = th.optim.Adam(lr=lr, params=mobilenetv3_large.parameters(), weight_decay=1e-4)

##### Hyperparameter search #####
#################################
parser = argparse.ArgumentParser(
                    prog='Traning CNNs',
                    description='Train the CNNs and Hyperparameter searching',
                    epilog='Current model MobileNetV3_Large')

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
  epoch_images = []
  train_losses = []

  for batch, labels in (pbar:= tqdm(train_loader, desc="batch", position=0)):
    # batch = batch.to(device)
    # labels = labels.to(device)

    optimizer.zero_grad(set_to_none=True) # optimizer.zero_grad()
    transformed = mobilenetv3_large.forward(batch)

    # Loss calculation
    loss = loss_fn(transformed, labels)

    # Metrics calculation
    # ADD METRIC
    
    # Backpropagation step
    loss.backward()
    optimizer.step()

    train_losses.append(loss)

    print(f"Epoch {1+epoch:5} - Loss: {loss.item():<7.5}")  # print(f"Epoch {1+epoch:5} - Loss: {loss.item():<7.5}")
    
    mobilenetv3_large.eval()
    with th.no_grad():
       val_losses = []
       result_images = []
       
       for batch, labels in val_loader:
         transformed = mobilenetv3_large.forward(batch)

         val_loss = loss_fn(transformed, labels) # change: labels:long to labels.type('torch.FloatTensor').to(device)
         val_losses.append(val_loss.item())
         print(f"Epoch {1+epoch:5} - Loss: {val_loss.item():<7.5}")

    # Metric calculation
    # ADD METRIC

      # if np.mean(test_losses) < best_metric_value:
      #       best_metric_value = np.mean(test_losses)
      #       best_model_state = model.state_dict()
      #       th.save(best_model_state, f"{results_dir}/{model_name}_epoch_{1+epoch:05}_{time.time()}.pt")
      #       grid = tv.utils.make_grid(epoch_images, nrow=7)
      #       tv.utils.save_image(grid, f"{results_dir}/{model_name}_epoch_{1+epoch:05}.png")
  
    wandb.log({
    "loss": np.mean(train_losses),
    # "Acc": np.mean(train_acc),
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



