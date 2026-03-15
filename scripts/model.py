import torch
import torch.nn as nn
import torchvision.models as models


class GSMNet(nn.Module):
    def __init__(self):
        super().__init__()

        base = models.resnet18(pretrained=True)
        num_features = base.fc.in_features

        base.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # ensures output is 0–1
        )

        self.model = base

    def forward(self, x):
        return self.model(x).squeeze(1)
