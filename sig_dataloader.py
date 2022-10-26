"""
taken and modified from https://github.com/pranv/ARC
"""

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
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split
import cv2
import scipy.ndimage as ndimage
from tqdm import tqdm

from module.preprocess import preprocess_signature, normalize_image

use_cuda = True

def imread_tool(img_path):
    image = np.asarray(Image.open(img_path).convert('L'))
    inputSize = image.shape
    normalized_image, cropped_image = normalize_image(image.astype(np.uint8), inputSize) # RETURN: normalized, cropped
    #image = normalized_image
    return Image.fromarray(normalized_image) , Image.fromarray(cropped_image)

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = cv2.bitwise_not(image)
    
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    image = cv2.erode(image, kernel)
    image = cv2.dilate(image, kernel)  
    """
    # skew correction
    """
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = 255 - cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    """
    image = Image.fromarray(image)
    return image

class SigDataset(Dataset):
    def __init__(self, path, train=True, image_size=256):
        self.path = path
        self.image_size = image_size
        self.basic_transforms = transforms.Compose([transforms.RandomInvert(1.0),
                                                    transforms.Resize((image_size,image_size)),
                                                    transforms.ToTensor(),
                                                    #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                                    #transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),
                                                    #transforms.Grayscale(num_output_channels=1)
                                                    ])
        self.augment_transforms = transforms.Compose([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
                                                    transforms.RandomErasing(),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomVerticalFlip(),
                                                    transforms.RandomResizedCrop((image_size,image_size)),
                                                    transforms.RandomInvert(),
                                                    ])
        data_root = path  # ./../ChiSig
        data_df = pd.DataFrame(columns=['img_path', 'label', 'writer_id'])

        self.img_path_dict = {}
        for img in os.listdir(data_root):
            img_split = img.split('-')
            self.img_path_dict.setdefault(img_split[0], []).append(img)

        self.img_dict = {}
        for writer_id, key in tqdm(enumerate(self.img_path_dict)):
            pos_label = self.img_path_dict[key][0].split('-')[1]
            pos_flag = False
            for img in self.img_path_dict[key]:
                img_split = img.split('-')
                if img_split[1] != pos_label:
                    pos_flag = True
            if pos_flag:
                for img in self.img_path_dict[key]:
                    img_path = os.path.join(data_root, img)
                    img_split = img.split('-')
                    label = int(img_split[1])
                    data_df = data_df.append({'img_path': img_path, 'label': label, 'writer_id': writer_id}, ignore_index=True)
                    sig_image, _ = imread_tool(img_path) # normalized
                    sig_image = self.basic_transforms(sig_image)
                    self.img_dict[img_path] = sig_image
            
        print(f'total {len(data_df)} images !!')
        # self.data_df = self.train_df = self.test_df = data_df
        self.data_df = data_df
        self.train_df, self.test_df = train_test_split(data_df, test_size=0.3, shuffle=False, random_state=1)
        data_df.to_csv("data.csv")
        #self.train_df = data_df[data_df['writer_id'] >= 100]
        #self.test_df = data_df[data_df['writer_id'] < 100]
        self.train_df.to_csv("data_train.csv")
        self.test_df.to_csv("data_test.csv")

        self.train = train

    def __len__(self):
        if self.train:
            return len(self.train_df)
        else:
            return len(self.test_df)

    def __getitem__(self, index):
        if self.train:
            self.group = self.train_df.groupby('writer_id')
            self.data_df = self.train_df
        else:
            self.group = self.test_df.groupby('writer_id')
            self.data_df = self.test_df
        
        group_writer_id = self.data_df.iloc[index]['writer_id']
        in_class_df = self.group.get_group(group_writer_id)
        # Anchor
        img_path = self.data_df.iloc[index]['img_path']
        #sig_image = Image.open(img_path).convert('RGB')
        #sig_image = imread_tool(img_path)
        #sig_image = self.basic_transforms(sig_image)
        sig_image = self.img_dict[img_path]
        #if self.train:
        #    sig_image = self.augment_transforms(sig_image)
        writer_id = self.data_df.iloc[index]['writer_id']
        label = self.data_df.iloc[index]['label']

        positive_path, negative_path = None, None

        # positive
        while True:
            #print(img_path, len(in_class_df[in_class_df['label'] == label]))
            if len(in_class_df[in_class_df['label'] == label]) == 1:
                positive_path = img_path
                break
            else:
                sample_df = in_class_df.sample()
                if sample_df['label'].item() == self.data_df.iloc[index]['label'] and sample_df['img_path'].item() != img_path:
                    positive_path = sample_df['img_path'].item()
                    #print(positive_path)
                    break
        #positive_sig_image = Image.open(positive_path).convert('RGB')
        #positive_sig_image = imread_tool(positive_path)
        #positive_sig_image = self.basic_transforms(positive_sig_image)
        positive_sig_image = self.img_dict[positive_path]
        #if self.train:
        #    positive_sig_image = self.augment_transforms(positive_sig_image)

        # negative
        while True:
            sample_df = in_class_df.sample()
            if sample_df['label'].item() != self.data_df.iloc[index]['label']:
                negative_path = sample_df['img_path'].item()
                #print(negative_path)
                break
        #negative_sig_image = Image.open(negative_path).convert('RGB')
        #negative_sig_image = imread_tool(negative_path)
        #negative_sig_image = self.basic_transforms(negative_sig_image)
        negative_sig_image = self.img_dict[negative_path]
        #if self.train:
        #    negative_sig_image = self.augment_transforms(negative_sig_image)
        
        image_pair_0 = torch.cat((sig_image, positive_sig_image), dim=0)
        if self.train: image_pair_0 = torch.squeeze(self.augment_transforms(torch.unsqueeze(image_pair_0, dim=1)))
        image_pair_1 = torch.cat((sig_image, negative_sig_image), dim=0)
        if self.train: image_pair_1 = torch.squeeze(self.augment_transforms(torch.unsqueeze(image_pair_1, dim=1)))
        image_pairs = torch.cat((torch.unsqueeze(image_pair_0, 0), torch.unsqueeze(image_pair_1, 0)), dim=0)
        #print(image_pairs.shape)

        if self.train:
            return image_pairs, torch.tensor([[1],[0]])
        else:
            return image_pairs, torch.tensor([[1],[0]]), {'Anchor':img_path, 'positive':positive_path, 'negative':negative_path}


class SigDataset_BH(Dataset):
    def __get_com_cropped__(self, image):
        '''image is a binary PIL image'''
        image = np.asarray(image)
        com = ndimage.measurements.center_of_mass(image)
        com = np.round(com)
        com[0] = np.clip(com[0], 0, image.shape[0])
        com[1] = np.clip(com[1], 0, image.shape[1])
        X_center, Y_center = int(com[0]), int(com[1])
        c_row, c_col = image[X_center, :], image[:, Y_center]

        x_start, x_end, y_start, y_end = -1, -1, -1, -1

        for i, v in enumerate(c_col):
            v = np.sum(image[i, :])
            if v < 255*image.shape[1]: # there exists text pixel
                if x_start == -1:
                    x_start = i
                else:
                    x_end = i

        for j, v in enumerate(c_row):
            v = np.sum(image[:, j])
            if v < 255*image.shape[0]: # there exists text pixel
                if y_start == -1:
                    y_start = j
                else:
                    y_end = j

        crop_rgb = Image.fromarray(np.asarray(image[x_start:x_end, y_start:y_end]))
        return crop_rgb
    
    def __init__(self, path, train=True, image_size=256, mode='normalized'):
        self.path = path
        self.image_size = image_size
        self.basic_transforms = transforms.Compose([transforms.RandomInvert(1.0),
                                                    transforms.Resize((image_size,image_size)),
                                                    transforms.ToTensor(),
                                                    #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                                    #transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),
                                                    #transforms.Grayscale(num_output_channels=1)
                                                    ])
        self.augment_transforms = transforms.Compose([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
                                                    transforms.RandomErasing(),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomVerticalFlip(),
                                                    #transforms.RandomResizedCrop((image_size,image_size)),
                                                    #transforms.RandomInvert(),
                                                    ])
        data_root = path  # ./../BHSig260/Bengali
        data_df = pd.DataFrame(columns=['img_path', 'label', 'writer_id'])

        self.img_dict = {}
        for dir in tqdm(os.listdir(data_root)):
            dir_path = os.path.join(data_root, dir)
            if os.path.isdir(dir_path):
                for img in os.listdir(dir_path):
                    img_path = os.path.join(dir_path, img)
                    img_split = img.split('-')
                    label = None
                    if img_split[3] == 'G':
                        label = 1
                    elif img_split[3] == 'F':
                        label = 0
                    assert label is not None
                    data_df = data_df.append({'img_path': img_path, 'label': label, 'writer_id': int(dir)}, ignore_index=True)
                    #sig_image = Image.open(img_path).convert('1')
                    # normalized, cropped
                    if mode == 'normalized':
                        sig_image, _ = imread_tool(img_path)
                    elif mode == 'cropped':
                        _, sig_image = imread_tool(img_path)
                        sig_image = sig_image.filter(ImageFilter.MinFilter(5))
                    elif mode == 'centered':
                        _, sig_image = imread_tool(img_path)
                        trans = transforms.CenterCrop(np.asarray(sig_image).shape[0])
                        sig_image = trans(sig_image)
                    elif mode == 'left':
                        sig_image, _ = imread_tool(img_path)
                        width, height = sig_image.size
                        left = 0
                        top = 0
                        right = width / 2
                        bottom = height
                        sig_image = sig_image.crop((left, top, right, bottom)).filter(ImageFilter.MinFilter(5))
                    else:
                        return NotImplementedError
                    #sig_image = self.__get_com_cropped__(sig_image)
                    sig_image = self.basic_transforms(sig_image)
                    self.img_dict[img_path] = sig_image

        print(f'total {len(data_df)} images !!')
        self.total_data_df = data_df
        pos_df = data_df.groupby('label').get_group(1)
        if 'Bengali' in path:
            #'''
            self.train_df = pos_df#[pos_df['writer_id'] >= 50]
            self.test_df = pos_df#[pos_df['writer_id'] < 50]
            '''
            self.train_df = pos_df[pos_df['writer_id'] >= 20]
            self.test_df = pos_df[pos_df['writer_id'] < 20]
            '''
            #self.train_df, self.test_df = train_test_split(pos_df, test_size=0.5, shuffle=False, random_state=1)
        elif 'Hindi' in path: # 100 train, 60 test
            self.train_df = pos_df[pos_df['writer_id'] >= 60]
            self.test_df = pos_df[pos_df['writer_id'] < 60]
            #self.train_df, self.test_df = train_test_split(pos_df, test_size=0.5, shuffle=False, random_state=1)
        data_df.to_csv("data.csv")

        self.train = train

    def __len__(self):
        if self.train:
            return len(self.train_df)
        else:
            return len(self.test_df)
    
    def __getitem__(self, index):
        if self.train:
            self.group = self.total_data_df.groupby('writer_id')
            self.data_df = self.train_df
        else:
            self.group = self.total_data_df.groupby('writer_id')
            self.data_df = self.test_df
        
        # Anchor
        img_path = self.data_df.iloc[index]['img_path']
        #print(img_path)
        #sig_image = imread_tool(img_path)
        sig_image = self.img_dict[img_path]
        #if self.train:
        #    sig_image = self.augment_transforms(sig_image)
        writer_id = self.data_df.iloc[index]['writer_id']
        label = self.data_df.iloc[index]['label']
        
        ###
        group_writer_id = self.data_df.iloc[index]['writer_id']
        in_class_df = self.group.get_group(group_writer_id)

        positive_path, negative_path = None, None

        # positive
        while True:
            #print(img_path, len(in_class_df[in_class_df['label'] == label]))
            if len(in_class_df[in_class_df['label'] == label]) == 1:
                positive_path = img_path
                break
            else:
                sample_df = in_class_df.sample()
                if sample_df['label'].item() == self.data_df.iloc[index]['label'] and sample_df['img_path'].item() != img_path:
                    positive_path = sample_df['img_path'].item()
                    break
        #print(positive_path)
        #positive_sig_image = Image.open(positive_path).convert('RGB')
        #positive_sig_image = imread_tool(positive_path)
        positive_sig_image = self.img_dict[positive_path]
        #if self.train:
        #    positive_sig_image = self.augment_transforms(positive_sig_image)

        # negative
        while True:
            sample_df = in_class_df.sample()
            if sample_df['label'].item() != self.data_df.iloc[index]['label']:
                negative_path = sample_df['img_path'].item()
                break
        #print(negative_path)
        #negative_sig_image = Image.open(negative_path).convert('RGB')
        #negative_sig_image = imread_tool(negative_path)
        negative_sig_image = self.img_dict[negative_path]
        #if self.train:
        #    negative_sig_image = self.augment_transforms(negative_sig_image)

        # negative 2
        '''
        while True:
            sample_df = in_class_df.sample()
            if sample_df['label'].item() != self.data_df.iloc[index]['label'] and sample_df['img_path'].item() != negative_path:
                negative_path_2 = sample_df['img_path'].item()
                break
        negative_sig_image_2 = self.img_dict[negative_path_2]
        '''
        
        if self.train:
            image_pair_0 = torch.cat((sig_image, positive_sig_image), dim=0)
            image_pair_0 = torch.squeeze(self.augment_transforms(torch.unsqueeze(image_pair_0, dim=1)))
            image_pair_1 = torch.cat((sig_image, negative_sig_image), dim=0)
            image_pair_1 = torch.squeeze(self.augment_transforms(torch.unsqueeze(image_pair_1, dim=1)))
            image_pair_2 = torch.cat((positive_sig_image, sig_image), dim=0)
            image_pair_2 = torch.squeeze(self.augment_transforms(torch.unsqueeze(image_pair_0, dim=1)))
            image_pair_3 = torch.cat((negative_sig_image, sig_image), dim=0)
            image_pair_3 = torch.squeeze(self.augment_transforms(torch.unsqueeze(image_pair_1, dim=1)))
            image_pairs = torch.cat((torch.unsqueeze(image_pair_0, 0),
                                    torch.unsqueeze(image_pair_1, 0),
                                    torch.unsqueeze(image_pair_2, 0),
                                    torch.unsqueeze(image_pair_3, 0)
                                    ), dim=0)
            #print(image_pairs.shape)
            return image_pairs, torch.tensor([[1],[0],[1],[0]])
        else:
            image_pair_0 = torch.cat((sig_image, positive_sig_image), dim=0)
            image_pair_1 = torch.cat((sig_image, negative_sig_image), dim=0)
            image_pairs = torch.cat((torch.unsqueeze(image_pair_0, 0), torch.unsqueeze(image_pair_1, 0)), dim=0)

            return image_pairs, torch.tensor([[1],[0]]), {'Anchor':img_path, 'positive':positive_path, 'negative':negative_path}

class single_test_dataset(Dataset):
    def __init__(self, anchor_path, positive_path, negative_path, image_size=256, mode='normalized'):
        self.anchor_path = anchor_path
        self.positive_path = positive_path
        self.negative_path = negative_path
        self.basic_transforms = transforms.Compose([transforms.RandomInvert(1.0),
                                                    transforms.Resize((image_size,image_size)),
                                                    transforms.ToTensor(),
                                                    #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                                    #transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),
                                                    #transforms.Grayscale(num_output_channels=1)
                                                    ])
        self.img_dict = {}
        path_dict = {'Anchor':anchor_path, 'positive':positive_path, 'negative':negative_path}
        for item in ['Anchor', 'positive', 'negative']:
            print(path_dict[item])
            # normalized, cropped
            if mode == 'normalized':
                sig_image, _ = imread_tool(path_dict[item])
                sig_image.save('img_pre_{}.png'.format(item))
            elif mode == 'cropped':
                _, sig_image = imread_tool(path_dict[item])
            elif mode == 'centered':
                _, sig_image = imread_tool(path_dict[item])
                trans = transforms.CenterCrop(np.asarray(sig_image).shape[0])
                sig_image = trans(sig_image)
            elif mode == 'left':
                sig_image, _ = imread_tool(path_dict[item])
                width, height = sig_image.size
                left = 0
                top = 0
                right = width / 2
                bottom = height
                sig_image = sig_image.crop((left, top, right, bottom)).filter(ImageFilter.MinFilter(5))
            else:
                return NotImplementedError
            sig_image = self.basic_transforms(sig_image)
            self.img_dict[item] = sig_image
    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        # 'Anchor', 'positive', 'negative'
        sig_image = self.img_dict['Anchor']
        positive_sig_image = self.img_dict['positive']
        negative_sig_image = self.img_dict['negative']
        image_pair_0 = torch.cat((sig_image, positive_sig_image), dim=0)
        image_pair_1 = torch.cat((sig_image, negative_sig_image), dim=0)
        image_pairs = torch.cat((torch.unsqueeze(image_pair_0, 0), torch.unsqueeze(image_pair_1, 0)), dim=0)

        return image_pairs, torch.tensor([[1],[0]]), {'Anchor':self.anchor_path, 'positive':self.positive_path, 'negative':self.negative_path}
