import torch
import torch.nn.functional as f
from torch import nn


class AquaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.norm4 = nn.BatchNorm2d(256)
        self.linear = nn.Linear(256, 64)
        self.dropout = nn.Dropout(p=0.2)
        self.final = nn.Linear(64, 19)

    def forward(self, x):
        x = self.pool(f.relu(self.norm1(self.conv1(x))))
        x = self.pool(f.relu(self.norm2(self.conv2(x))))
        x = self.pool(f.relu(self.norm3(self.conv3(x))))
        x = f.relu(self.norm4(self.conv4(x)))

        # Change x from shape [batch_size, channels, h, w] to [batch_size, channels, h * w]
        x = x.view(x.shape[0], x.shape[1], -1)

        x = x.mean(-1)

        x = f.relu(self.linear(x))
        x = self.dropout(x)

        x = self.final(x)

        return x


net = AquaNet()
x = net(torch.randn([3, 3, 64, 64]))
print(x.shape)
