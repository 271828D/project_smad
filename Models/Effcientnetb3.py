import torch as th
import torch as th
import torch.nn as nn
import torchvision as tv


class EfficientnetB3(nn.Module):
    """Pretrained model from Pytorch: https://pytorch.org/vision/main/models/efficientnet.html"""
    def __init__(self, in_channels, out_channels):
        super(EfficientnetB3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.model = tv.models.efficientnet_b3(weights = 'IMAGENET1K_V1', progress = True)
        
        self.lastlayer = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        # return self.lastlayer(x)
        return x