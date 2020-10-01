import torch.nn as nn
import torch.nn.functional as F


class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.norm1 = nn.BatchNorm2d(3)
        self.norm2 = nn.BatchNorm2d(8)
        self.norm3 = nn.BatchNorm2d(16)
        self.norm4 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(64 * 24 * 24, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(self.conv1(F.relu(self.norm1(x))))
        x = self.pool(self.conv2(F.relu(self.norm2(x))))
        x = self.pool(self.conv3(F.relu(self.norm3(x))))
        x = self.pool(self.conv4(F.relu(self.norm4(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x