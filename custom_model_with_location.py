import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomModelLoc(nn.Module):
  def __init__(self, num_classes):
    super(CustomModelLoc, self).__init__()
    self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
    self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
    self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
    self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.norm1 = nn.BatchNorm2d(3)
    self.norm2 = nn.BatchNorm2d(8)
    self.norm3 = nn.BatchNorm2d(16)
    self.norm4 = nn.BatchNorm2d(32)
    self.fc1 = nn.Linear(64* 24* 24, 128)
    self.fc2 = nn.Linear(132, 64)
    self.fc3 = nn.Linear(64, num_classes)

  def forward(self, img, data):
    img = self.pool(self.conv1(F.relu(self.norm1(img))))
    img = self.pool(self.conv2(F.relu(self.norm2(img))))
    img = self.pool(self.conv3(F.relu(self.norm3(img))))
    img = self.pool(self.conv4(F.relu(self.norm4(img))))
    img = img.view(img.size(0), -1)
    img = F.relu(self.fc1(img))
    out = torch.cat((img, data), 1)
    out = F.relu(self.fc2(out))
    out = self.fc3(out)
    return out