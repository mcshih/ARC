from builtins import print
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class SpatialTransformerNetwork(nn.Module):
    def __init__(self, I_size, I_r_size, I_channel_num=1, padding_mode="zeros"):
        super(SpatialTransformerNetwork, self).__init__()
        self.I_size = I_size
        self.I_r_size = I_r_size  # = (I_r_height, I_r_width)
        self.I_channel_num = I_channel_num
        self.padding_mode = padding_mode # 'zeros', 'border'
        self.local_net = LocalNetwork(I_size, I_r_size, I_channel_num, padding_mode)
    def forward(self, img):
        transform_img = self.local_net(img)
        return transform_img
    def save_to_file(self, file_path: str) -> None:
        torch.save(self.state_dict(), file_path)


class LocalNetwork(nn.Module):
    def __init__(self, I_size, I_r_size, I_channel_num=1, padding_mode="zeros"):
        super(LocalNetwork, self).__init__()
        self.I_size = I_size
        self.I_r_size = I_r_size  # = (I_r_height, I_r_width)
        self.I_channel_num = I_channel_num
        self.padding_mode = padding_mode # 'zeros', 'border'
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.I_channel_num, out_channels=64, kernel_size=3, stride=1, padding=1,
                      bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 64 x I_height/2 x I_width/2
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 128 x I_height/4 x I_width/4
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 256 x I_height/8 x I_width/8
            nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)  # batch_size x 512
        )

        self.localization_fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
        self.localization_fc2 = nn.Linear(256, 6)
        bias = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0]))

        nn.init.constant_(self.localization_fc2.weight, 0)
        self.localization_fc2.bias.data.copy_(bias)

    def forward(self, img):
        '''

        :param img: (b, c, h, w)
        :return: (b, c, h, w)
        '''
        batch_size = img.size(0)
        img = torch.cat((img[0::2], img[1::2]), 1)
        features = self.conv(img).view(batch_size, -1)
        batch_C_prime = torch.repeat_interleave(batch_C_prime, 2, dim=0)
        theta = self.localization_fc2(self.localization_fc1(features)).view(batch_size, 2, 3)
        
        grid = F.affine_grid(theta, torch.Size((batch_size, self.I_channel_num, self.I_r_size[0], self.I_r_size[1])))
        img_transform = F.grid_sample(img, grid, padding_mode=self.padding_mode)

        return img_transform, theta


if __name__ == '__main__':
    net = LocalNetwork()

    x = torch.randn(1, 1, 40, 40) + 1
    net(x)
