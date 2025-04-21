import torch
import torch.nn as nn
import torch.nn.functional as F

class ShallowConvNet(nn.Module):
    def __init__(self, num_classes, time_series_size, num_kernels, node_size):
        super(ShallowConvNet, self).__init__()
        self.num_kernels = num_kernels
        self.time_series_size = time_series_size
        self.num_classes = num_classes
        self.node_size = node_size

        self.features = nn.Sequential(
            nn.Conv2d(1, self.num_kernels, (1, 25)),
            nn.Conv2d(self.num_kernels, self.num_kernels, (self.node_size, 1), bias=False),
            nn.BatchNorm2d(self.num_kernels)
        )
        self.avgpool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.dropout = nn.Dropout()
        hidden_size = (((self.time_series_size - 25 + 1) - 75) // 15 + 1) * self.num_kernels
        self.classifier = nn.Linear(hidden_size, self.num_classes)

    def forward(self, time_series):
        # time_series = time_series.unsqueeze(1)
        hidden_state = self.features(time_series)
        hidden_state = torch.square(hidden_state)
        hidden_state = self.avgpool(hidden_state)
        hidden_state = torch.clip(torch.log(hidden_state), min=1e-7, max=1e7)
        hidden_state = self.dropout(hidden_state)
        # hidden_state = hidden_state.mean(-1)
        features = torch.flatten(hidden_state, 1)
        logits = self.classifier(features)

        return logits
