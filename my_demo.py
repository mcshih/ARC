import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
from module.transformation import TPS_SpatialTransformerNetwork

def __getpatches__(x_arr):
    ptsz = 16
    patches = []
    Batch_size ,C, H, W = x_arr.shape # 1, 256, 256
    num_H = 2
    num_W = 2

    for i in range(num_H):
        for j in range(num_W):
            start_x = i*ptsz
            end_x = start_x + ptsz
            start_y = j*ptsz
            end_y = start_y + ptsz

            patch = x_arr[:,:, start_x:end_x, start_y:end_y]
            patches.append(torch.unsqueeze(patch, 0))

    return torch.squeeze(torch.cat(patches, dim=0), 1)

"""
data = np.load('data/omniglot.npy')
data_1 = np.load('data/images_background.npy')
data_2 = np.load('data/images_evaluation.npy')
print(data.shape, data_1.shape, data_1.shape)
single = data[0][0]
print(single.shape)
plt.imshow(single, cmap="gray")
plt.savefig('my_depo.png')
plt.close()
"""
chars = np.load('data/omniglot.npy')

image_size = 32
device = torch.device("cuda")
net = TPS_SpatialTransformerNetwork(F=20, I_size=(image_size, image_size), I_r_size=(image_size, image_size), I_channel_num=1)
net.train()
net.to(device)

resized_chars = np.zeros((1623, 20, image_size, image_size), dtype='uint8')
for i in range(1623):
    for j in range(20):
        resized_chars = torch.tensor(resize(chars[i, j], (image_size, image_size)) *255)
        resized_chars = torch.unsqueeze(torch.unsqueeze(resized_chars, 0),0).to(torch.float).to(device)
        print(resized_chars.shape)
        output = __getpatches__(net(resized_chars))
        print(output.shape)
        break
    break