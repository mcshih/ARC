# %%
from builtins import print
import os
import numpy as np
from numpy.random import choice
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from skimage.transform import resize
import torchvision.transforms as transforms
from image_augmenter import ImageAugmenter
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2
import scipy.ndimage as ndimage
from tqdm import tqdm
import matplotlib.pyplot as plt
from module.preprocess import preprocess_signature, normalize_image

# %%
# ./../BHSig260/Hindi/058/H-S-58-G-03.tif

# %%
img_path = "./../BHSig260/Hindi/058/H-S-58-G-03.tif"
image = np.asarray(Image.open(img_path).convert('L'))
inputSize = image.shape
image = normalize_image(image.astype(np.uint8), inputSize)

im = Image.fromarray(image)
im.save("my_deom.png")
