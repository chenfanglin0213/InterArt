# _*_ coding : utf-8 _*_
# @Time : 2022/10/11 9:05
# @Author : chenfanglin
# @File : model.py
# @Project : InterArt

import torch
import torch.nn as nn

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1)
]
# CNNBlock
# FC


class CNNBlock(nn.Module):
    def __init__(self, in_channel, out_channel, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, bias = False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channel)
        # 激活函数
        self.lk = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.lk(self.bn(self.conv(x)))


class YoloV1(nn.Module):
    def __init__(self, in_channel, S, B, C):
        super(YoloV1, self).__init__()
        self.in_channel = in_channel
        self.conv = self._make_conv()
        self.fc = self._make_fc(grid_size=S, num_box=B, num_class=C)

    def forward(self, x):
        return self.fc(self.conv(x))

    def _make_conv(self):
        # use architecture_config
        layers = []
        for config in architecture_config:
            # 三种情况
            if type(config) == tuple:
                pass
                conv = CNNBlock(self.in_channel, config[1], kernel_size=config[0], stride=config[2], padding=config[3])
                layers.append(conv)
                self.in_channel = config[1]
            elif type(config) == list:
                conv1_config = config[0]
                conv2_config = config[1]
                times = config[2]
                for _ in range(times):
                    # repeate 多少次
                    conv1 = CNNBlock(self.in_channel, conv1_config[1], kernel_size=conv1_config[0], stride=conv1_config[2], padding=conv1_config[3])
                    layers.append(conv1)
                    conv2 = CNNBlock(conv1_config[1], conv2_config[1], kernel_size=conv2_config[0], stride=conv2_config[2], padding=conv2_config[3])
                    layers.append(conv2)

                # end for
                self.in_channel = conv2_config[1]
            elif type(config) == "M":
                maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
                layers.append(maxpool)

        return nn.Sequential(*layers)

    def _make_fc(self, grid_size, num_box, num_class):
        # grid_size = S, num_box = B, num_class = C
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(grid_size * grid_size * 1024, 4096),
            nn.Dropout(),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, grid_size * grid_size * (num_box * 5 + num_class))
        )


input = torch.randn(32, 3, 224, 224)
yolo = YoloV1(3, 7, 2, 20)
print(yolo(input).shape)
