import torch.nn as nn
from torchvision import datasets, models

def resnet152(num_classes: int) -> nn.Module:
    model_ft = models.resnet152(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft
