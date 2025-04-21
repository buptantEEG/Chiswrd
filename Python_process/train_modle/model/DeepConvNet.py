import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepConvNet(nn.Module):

    def __init__(self, num_classes, time_series_size, num_kernels, node_size):
        super(DeepConvNet, self).__init__()
        self.num_kernels = num_kernels
        self.time_series_size = time_series_size
        self.num_classes = num_classes
        self.node_size = node_size

        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.num_kernels, (1, 5)),
            nn.Conv2d(self.num_kernels, self.num_kernels, (self.node_size, 1), bias=False),
            nn.BatchNorm2d(self.num_kernels),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(0.5)
        )
        hidden_size = (((self.time_series_size - 5 + 1) - 2) // 2 + 1)
        self.block2 = nn.Sequential(
            nn.Conv2d(self.num_kernels, self.num_kernels*2, (1, 5)),
            nn.BatchNorm2d(self.num_kernels*2),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(0.5)
        )
        hidden_size = (((hidden_size - 5 + 1) - 2) // 2 + 1)
        self.block3 = nn.Sequential(
            nn.Conv2d(self.num_kernels*2, self.num_kernels*4, (1, 5)),
            nn.BatchNorm2d(self.num_kernels*4),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(0.5)
        )
        hidden_size = (((hidden_size - 5 + 1) - 2) // 2 + 1) * self.num_kernels * 4
        # hidden_size = self.num_kernels * 4  # ZuCo
        self.classifier = nn.Linear(hidden_size, self.num_classes)

    def forward(self, time_series):
        # time_series = time_series.unsqueeze(1)
        hidden_state = self.block1(time_series)
        hidden_state = self.block2(hidden_state)
        hidden_state = self.block3(hidden_state)
        # hidden_state = hidden_state.mean(-1)
        features = torch.flatten(hidden_state, 1)
        logits = self.classifier(features)
        return logits
