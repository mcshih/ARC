from builtins import print
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class SpatialTransformerNetwork(nn.Module):
    def __init__(self, I_size, I_r_size, I_channel_num=1, padding_mode="zeros", box_num = 1, boxSize = 2):
        super(SpatialTransformerNetwork, self).__init__()
        self.I_size = I_size
        self.I_r_size = I_r_size
        self.I_channel_num = I_channel_num
        self.padding_mode = padding_mode # 'zeros', 'border'
        if box_num == 1:
            self.local_net = LocalNetwork(I_size, I_r_size, I_channel_num, padding_mode)
        elif box_num > 1:
            self.local_net = LocalNetwork_mul(I_size, I_r_size, I_channel_num, padding_mode, box_num, boxSize)
        else:
            return NotImplementedError
    def forward(self, img):
        transform_img = self.local_net(img)
        #print(transform_img.shape)
        return transform_img
    def save_to_file(self, file_path: str) -> None:
        torch.save(self.state_dict(), file_path)


class LocalNetwork(nn.Module):
    def __init__(self, I_size, I_r_size, I_channel_num=1, padding_mode="zeros"):
        super(LocalNetwork, self).__init__()
        self.I_size = I_size
        self.I_r_size = I_r_size
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
        self.localization_fc2 = nn.Linear(256, 2)
        #bias = torch.from_numpy(np.array([2., 0, 0, 0, 2., 0]))
        bias = torch.from_numpy(np.array([0., 0.]))

        nn.init.constant_(self.localization_fc2.weight, 0)
        self.localization_fc2.bias.data.copy_(bias)

    def forward(self, img):
        '''

        :param img: (b, c, h, w)
        :return: (b, c, h, w)
        '''
        batch_size = img.size(0)
        features = self.conv(img[0::4]).view(batch_size//4, -1)
        theta = self.localization_fc2(self.localization_fc1(features))#.view(batch_size, 2, 3)
        theta = torch.repeat_interleave(theta, 4, dim=0)

        input_theta_1 = torch.stack((0.5*torch.ones(batch_size).cuda(), torch.zeros(batch_size).cuda(), theta.T[0]))
        input_theta_2 = torch.stack((torch.zeros(batch_size).cuda(), 0.5*torch.ones(batch_size).cuda(), theta.T[1]))
        input_theta = torch.stack((input_theta_1.T, input_theta_2.T), dim=1)
        #print(input_theta_1.shape, input_theta.shape, input_theta[0])
        grid = F.affine_grid(input_theta, torch.Size((batch_size, self.I_channel_num, self.I_r_size[0], self.I_r_size[1])))
        img_transform = F.grid_sample(img, grid, padding_mode=self.padding_mode)

        return img_transform

class LocalNetwork_mul(nn.Module):
    def __init__(self, I_size, I_r_size, I_channel_num=1, padding_mode="zeros", box_num = 3, boxSize = 2):
        super(LocalNetwork_mul, self).__init__()
        self.I_size = I_size
        self.I_r_size = I_r_size
        self.I_channel_num = I_channel_num
        self.padding_mode = padding_mode # 'zeros', 'border'
        self.box_num = box_num
        self.boxSize = boxSize
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
        self.localization_fc2 = nn.Linear(256, 2 * box_num)
        bias = torch.from_numpy(np.zeros(2 * box_num))

        nn.init.constant_(self.localization_fc2.weight, 0)
        self.localization_fc2.bias.data.copy_(bias)

    def forward(self, img):
        '''

        :param img: (b, c, h, w)
        :return: (b, c, h, w)
        '''
        batch_size = img.size(0)
        features = self.conv(img[0::4]).view(batch_size//4, -1)
        theta = self.localization_fc2(self.localization_fc1(features))#.view(batch_size, 2, 3)
        theta = theta.view(batch_size//4 * self.box_num, 2)
        theta = torch.repeat_interleave(theta, 4, dim=0)
        #print(theta.shape)

        # 0.5 for 2X, 0.25 for 4X
        input_theta_1 = torch.stack(((1.0/self.boxSize)*torch.ones(batch_size * self.box_num).cuda(), torch.zeros(batch_size * self.box_num).cuda(), theta.T[0]))
        input_theta_2 = torch.stack((torch.zeros(batch_size * self.box_num).cuda(), (1.0/self.boxSize)*torch.ones(batch_size* self.box_num).cuda(), theta.T[1]))
        input_theta = torch.stack((input_theta_1.T, input_theta_2.T), dim=1)
        #print(input_theta_1.shape, input_theta.shape, input_theta[0])
        grid = F.affine_grid(input_theta, torch.Size((batch_size * self.box_num, self.I_channel_num, self.I_r_size[0], self.I_r_size[1])))
        img = img.repeat(self.box_num, 1, 1, 1)
        img_transform = F.grid_sample(img, grid, padding_mode=self.padding_mode)

        return img_transform

if __name__ == '__main__':
    net = LocalNetwork()

    x = torch.randn(1, 1, 40, 40) + 1
    net(x)
