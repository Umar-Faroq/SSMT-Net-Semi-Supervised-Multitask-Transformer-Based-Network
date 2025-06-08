#!/usr/bin/env python
# coding: utf-8

# ## Code here onward

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'
import cv2
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import torchvision.transforms as transforms
from glob import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Function to Load Image
def load_image(path, size):
    """Loads an image, resizes it, converts to grayscale, and normalizes."""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Directly load as grayscale
    image = cv2.resize(image, (size, size))
    image = image / 255.0  # Normalize to [0,1]
    return image

# Function to Load Data
def load_data(image_folder, mask_folder, size):
    """Loads images and corresponding masks, and computes normalized nodule areas."""
    images, masks, areas = [], [], []
    
    # Get sorted file paths
    image_paths = sorted(glob(os.path.join(image_folder, "*")))
    mask_paths = sorted(glob(os.path.join(mask_folder, "*")))

    # Create dictionaries mapping filenames to paths
    image_dict = {os.path.basename(p): p for p in image_paths}
    mask_dict = {os.path.basename(p): p for p in mask_paths}

    for filename in image_dict.keys():
        if filename in mask_dict:  # Ensure mask exists for this image
            img = load_image(image_dict[filename], size)
            mask = load_image(mask_dict[filename], size)

            # Compute nodule area (sum of mask pixels)
            area = np.sum(mask)

            images.append(img)
            masks.append(mask)
            areas.append(area)

    # Normalize the areas to range [0, 1]
    areas = np.array(areas)
    min_area, max_area = areas.min(), areas.max()
    normalized_areas = (areas - min_area) / (max_area - min_area + 1e-8)  # Avoid division by zero

    return np.array(images), np.array(masks), normalized_areas

# Define paths
trainval_image_folder = "Thyroid Dataset/tn3k/trainval-image"
trainval_mask_folder = "Thyroid Dataset/tn3k/trainval-mask"
test_image_folder = "Thyroid Dataset/tn3k/test-image"
test_mask_folder = "Thyroid Dataset/tn3k/test-mask"

# Set image size
img_size = 224

# Load trainval and test data separately
X_trainval, Y_trainval, area_trainval = load_data(trainval_image_folder, trainval_mask_folder, img_size)
X_test, Y_test, area_test = load_data(test_image_folder, test_mask_folder, img_size)

# Split trainval into training (80%) and validation (20%)
X_train, X_valid, Y_train, Y_valid, area_train, area_valid = train_test_split(
    X_trainval, Y_trainval, area_trainval, test_size=0.2, shuffle=True, random_state=42
)

# Normalize the areas
scaler = MinMaxScaler()
area_trainval = area_trainval.reshape(-1, 1)  # Fit only on trainval data
scaler.fit(area_trainval)
area_train = scaler.transform(area_train.reshape(-1, 1)).flatten()
area_valid = scaler.transform(area_valid.reshape(-1, 1)).flatten()
area_test = scaler.transform(area_test.reshape(-1, 1)).flatten()

def prepare_data(X, Y):
    X = np.expand_dims(X, axis=-1)  # Add channel dimension
    Y = np.expand_dims(Y, axis=-1)  # Add channel dimension
    X = torch.from_numpy(X).float().permute(0, 3, 1, 2)  # Convert to (N, C, H, W)
    Y = torch.from_numpy(Y).float().permute(0, 3, 1, 2)
    return X, Y

X_train, Y_train = prepare_data(X_train, Y_train)
X_valid, Y_valid = prepare_data(X_valid, Y_valid)
X_test, Y_test = prepare_data(X_test, Y_test)
class RandomZoomOut:
    """Randomly zooms out an image and its corresponding mask by padding and resizing back."""
    def __init__(self, scale=(1.0, 1.3), p=0.2):
        self.scale = scale
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            w, h = img.size
            scale_factor = random.uniform(*self.scale)
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)

            # Create new blank images (black padding for image and mask)
            new_img = Image.new("L", (new_w, new_h), 0)
            new_mask = Image.new("L", (new_w, new_h), 0)

            x_offset = (new_w - w) // 2
            y_offset = (new_h - h) // 2

            # Paste the original image and mask onto the new canvas
            new_img.paste(img, (x_offset, y_offset))
            new_mask.paste(mask, (x_offset, y_offset))

            # Resize back to original size
            img = new_img.resize((w, h), Image.BILINEAR)
            mask = new_mask.resize((w, h), Image.NEAREST)  # Nearest for masks

        return img, mask

class RandomMergeImages:
    """Randomly merges two images and masks side by side 30% of the time."""
    def __init__(self, dataset, p=0.3):
        self.dataset = dataset  # The dataset to sample the second image from
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            idx = random.randint(0, len(self.dataset) - 1)
            second_img, second_mask = self.dataset[idx]  # Get a random image-mask pair

            # Convert tensor to PIL image if necessary
            if isinstance(second_img, torch.Tensor):
                second_img = transforms.ToPILImage()(second_img)
            if isinstance(second_mask, torch.Tensor):
                second_mask = transforms.ToPILImage()(second_mask)

            # Resize to match the input image size
            second_img = second_img.resize(img.size)
            second_mask = second_mask.resize(mask.size)

            # Merge images and masks side by side
            merged_img = Image.new("L", (img.width * 2, img.height))  # Grayscale
            merged_mask = Image.new("L", (mask.width * 2, mask.height))  # Grayscale

            merged_img.paste(img, (0, 0))
            merged_img.paste(second_img, (img.width, 0))

            merged_mask.paste(mask, (0, 0))
            merged_mask.paste(second_mask, (mask.width, 0))

            # Resize back to original dimensions
            img = merged_img.resize(img.size, Image.BILINEAR)
            mask = merged_mask.resize(mask.size, Image.NEAREST)  # Nearest for masks

        return img, mask

import torch
import random
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
import os
import cv2
import random
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Function to Load Image
def load_image(path, size):
    """Loads an image, resizes it, converts to grayscale, and normalizes."""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Directly load as grayscale
    image = cv2.resize(image, (size, size))
    image = image / 255.0  # Normalize to [0,1]
    return image

# Function to Load Data
def load_data(image_folder, mask_folder, size):
    """Loads images and corresponding masks, and computes normalized areas."""
    images, masks, areas = [], [], []
    
    image_paths = sorted(glob(os.path.join(image_folder, "*")))
    mask_paths = sorted(glob(os.path.join(mask_folder, "*")))

    image_dict = {os.path.basename(p): p for p in image_paths}
    mask_dict = {os.path.basename(p): p for p in mask_paths}

    for filename in image_dict.keys():
        if filename in mask_dict:  # Ensure mask exists for this image
            img = load_image(image_dict[filename], size)
            mask = load_image(mask_dict[filename], size)
            area = np.sum(mask)
            
            images.append(img)
            masks.append(mask)
            areas.append(area)

    areas = np.array(areas)
    min_area, max_area = areas.min(), areas.max()
    normalized_areas = (areas - min_area) / (max_area - min_area + 1e-8)  # Avoid division by zero

    return np.array(images), np.array(masks), normalized_areas

# Define dataset class
class NoduleGlandDataset(Dataset):
    """Dataset for Nodule and Gland images and masks with augmentations."""
    
    def __init__(self, nodule_data, gland_data, augment=False):
        self.nodule_data = nodule_data
        self.gland_data = gland_data
        self.augment = augment

        self.base_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.15),
            transforms.RandomVerticalFlip(p=0.15),
            transforms.RandomRotation(10, interpolation=transforms.InterpolationMode.NEAREST),
        ])

        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.mask_transform = transforms.ToTensor()
    
    def __len__(self):
        return len(self.nodule_data[0])

    def __getitem__(self, idx):
        """Loads the image and mask using their file paths."""
        nodule_img, nodule_mask, nodule_area = self.nodule_data[0][idx], self.nodule_data[1][idx], self.nodule_data[2][idx]
        gland_img, gland_mask, gland_area = self.gland_data[0][idx], self.gland_data[1][idx], self.gland_data[2][idx]
        
        # Convert NumPy array to PIL Image before applying transforms
        nodule_img = Image.fromarray(nodule_img)
        nodule_mask = Image.fromarray(nodule_mask)
        gland_img = Image.fromarray(gland_img)
        gland_mask = Image.fromarray(gland_mask)
        
        if self.augment:
            # Apply the same base augmentations to both image and mask
            seed = random.randint(0, 9999)
            torch.manual_seed(seed)
            nodule_img = self.base_transform(nodule_img)
            gland_img = self.base_transform(gland_img)

            torch.manual_seed(seed)
            nodule_mask = self.base_transform(nodule_mask)
            gland_mask = self.base_transform(gland_mask)
        
        # Convert image and mask to tensor
        nodule_img = self.tensor_transform(nodule_img)
        nodule_mask = self.mask_transform(nodule_mask)
        gland_img = self.tensor_transform(gland_img)
        gland_mask = self.mask_transform(gland_mask)
        
        return nodule_img, nodule_mask, torch.tensor(nodule_area, dtype=torch.float32), gland_img, gland_mask, torch.tensor(gland_area, dtype=torch.float32)

# Define paths for datasets
trainval_image_folder = "Thyroid Dataset/tn3k/trainval-image"
trainval_mask_folder = "Thyroid Dataset/tn3k/trainval-mask"
test_image_folder = "Thyroid Dataset/tn3k/test-image"
test_mask_folder = "Thyroid Dataset/tn3k/test-mask"

train_gland_img_folder = "Thyroid Dataset/tg3k/thyroid-image"
train_gland_mask_folder = "Thyroid Dataset/tg3k/thyroid-mask"

img_size = 224

# Load datasets
X_trainval, Y_trainval, area_trainval = load_data(trainval_image_folder, trainval_mask_folder, img_size)
X_test, Y_test, area_test = load_data(test_image_folder, test_mask_folder, img_size)
X_gland, Y_gland, area_gland = load_data(train_gland_img_folder, train_gland_mask_folder, img_size)

# Split trainval into training (80%) and validation (20%)
X_train, X_valid, Y_train, Y_valid, area_train, area_valid = train_test_split(
    X_trainval, Y_trainval, area_trainval, test_size=0.2, shuffle=True, random_state=42
)

# Create dataset and dataloaders
train_dataset = NoduleGlandDataset((X_train, Y_train, area_train), (X_gland, Y_gland, area_gland), augment=True)
valid_dataset = NoduleGlandDataset((X_valid, Y_valid, area_valid), (X_gland, Y_gland, area_gland), augment=False)
test_dataset = NoduleGlandDataset((X_test, Y_test, area_test), (X_gland, Y_gland, area_gland), augment=False)

# Create DataLoaders
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# **vit_seg_configs.py**

# In[2]:


import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import ml_collections

import math

from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F



import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_testing():
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.n_skip = 3
    config.activation = 'softmax'

    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.patches.size = (32, 32)
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_32.npz'
    return config


def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.representation_size = None

    # custom
    config.classifier = 'seg'
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-L_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_r50_l16_config():
    """Returns the Resnet50 + ViT-L/16 configuration. customized """
    config = get_l16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.resnet_pretrained_path = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.patches.size = (32, 32)
    return config


def get_h14_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (14, 14)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.num_heads = 16
    config.transformer.num_layers = 32
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None

    return config


# **vit_seg_modeling_resnet_skip.py**

# In[3]:


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))

class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]


# **vit_seg_modeling.py**

# In[4]:


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

CONFIGS = {
    'ViT-B_16': get_b16_config(),
    'ViT-B_32': get_b32_config(),
    'ViT-L_16': get_l16_config(),
    'ViT-L_32': get_l32_config(),
    'ViT-H_14': get_h14_config(),
    'R50-ViT-B_16': get_r50_b16_config(),
    'R50-ViT-L_16': get_r50_l16_config(),
    'testing': get_testing(),
}


# ## model with HQ reconstruction

# In[5]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module.
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
        num_out_ch (int): Number of output channels.
        input_resolution (tuple[int, int], optional): Resolution of input image.
    """
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

class ReconstructionDecoder(nn.Module):
    def __init__(self, embed_dim, num_feat, num_out_ch, upsampler, upscale, patches_resolution):
        super(ReconstructionDecoder, self).__init__()
        self.upsampler = upsampler
        self.upscale = upscale
        self.patches_resolution = patches_resolution

        if upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif upsampler == 'nearest+conv':
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if upscale == 4:
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

    def forward(self, x):
        if self.upsampler == 'pixelshuffle':
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(F.interpolate(x, scale_factor=2, mode='nearest')))
            if self.upscale == 4:
                x = self.lrelu(self.conv_up2(F.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            x = self.conv_last(x)
        return x
class ReconstructionDecoder(nn.Module):
    def __init__(self, embed_dim, num_feat, num_out_ch, upsampler, upscale, patches_resolution, img_size):
        super(ReconstructionDecoder, self).__init__()
        self.upsampler = upsampler
        self.upscale = upscale
        self.patches_resolution = patches_resolution
        self.embed_dim = embed_dim
        self.num_feat = num_feat
        self.img_size = img_size  # Store original image size

        if upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        else:
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

    def forward(self, x):
        # Reshape input (B, N, C) -> (B, C, H, W)
        B, N, C = x.shape
        H, W = self.patches_resolution

        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)

        if self.upsampler == 'pixelshuffle':
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        else:
            x = self.conv_last(x)

        # Ensure the final output matches the original image size
        x = F.interpolate(x, size=self.img_size, mode="bilinear", align_corners=False)

        return x


class VisionTransformerWithReconstruction(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformerWithReconstruction, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )

        # Pass original image size to reconstruction decoder
        self.reconstruction_decoder = ReconstructionDecoder(
            embed_dim=config.hidden_size, num_feat=64, num_out_ch=config['n_classes'],
            upsampler='pixelshuffle', upscale=2, patches_resolution=(img_size // 16, img_size // 16),
            img_size=(img_size, img_size)  # Ensure it restores to the original input size
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        x, attn_weights, features = self.transformer(x)

        # Ensure the reconstruction output matches input dimensions
        reconstruction_output = self.reconstruction_decoder(x)

        segmentation_output = self.decoder(x, features)
        logits = self.segmentation_head(segmentation_output)

        return logits, reconstruction_output

import torch
import torch.nn as nn

class VisionTransformerWithDualSegmentation(nn.Module):
    def __init__(self, config, img_size=224, num_classes=1, zero_head=False, vis=False):
        super(VisionTransformerWithDualSegmentation, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        
        # First segmentation decoder
        self.seg_decoder1 = DecoderCup(config)
        self.segmentation_head1 = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        
        # Second segmentation decoder
        self.seg_decoder2 = DecoderCup(config)
        self.segmentation_head2 = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        
        # Reconstruction decoder
        self.reconstruction_decoder = ReconstructionDecoder(
            embed_dim=config.hidden_size, num_feat=64, num_out_ch=config['n_classes'],
            upsampler='pixelshuffle', upscale=2, patches_resolution=(img_size // 16, img_size // 16),
            img_size=(img_size, img_size)
        )
        
        # Fully connected layer for nodule area prediction (deeper network)
        self.nodule_area_fc = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output a single value for nodule area
        )
        
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        x, attn_weights, features = self.transformer(x)
        
        # First segmentation output
        seg_output1 = self.seg_decoder1(x, features)
        logits1 = self.segmentation_head1(seg_output1)
        
        # Second segmentation output
        seg_output2 = self.seg_decoder2(x, features)
        logits2 = self.segmentation_head2(seg_output2)
        
        # Reconstruction output
        reconstruction_output = self.reconstruction_decoder(x)
        
        # Nodule area prediction
        nodule_area = self.nodule_area_fc(x[:, 0, :])  # Using the CLS token representation
        
        return logits1, logits2, reconstruction_output, nodule_area


def get_dual_segmentation_network(vit_name='R50-ViT-B_16', img_size=224, num_classes=1, n_skip=3, vit_patches_size=16):
    """
    Initializes the Vision Transformer-based segmentation network with dual segmentation outputs,
    one reconstruction output, and a fully connected layer for nodule area prediction.
    """
    config_vit = CONFIGS[vit_name]
    config_vit.n_classes = num_classes
    config_vit.n_skip = n_skip
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (img_size // vit_patches_size, img_size // vit_patches_size)

    net = VisionTransformerWithDualSegmentation(
        config=config_vit, 
        img_size=img_size, 
        num_classes=num_classes
    )

    return net


# In[6]:


def get_network(vit_name='R50-ViT-B_16', img_size=224, num_classes=2, n_skip=3, vit_patches_size=16):
    """
    Initializes the Vision Transformer-based segmentation network with added HQ Reconstruction.
    """
    config_vit = CONFIGS[vit_name]
    config_vit.n_classes = num_classes
    config_vit.n_skip = n_skip
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (img_size // vit_patches_size, img_size // vit_patches_size)

    net = VisionTransformerWithReconstruction(
        config=config_vit, 
        img_size=img_size, 
        num_classes=num_classes
    )

    return net


# Training Function
import torch.optim.lr_scheduler as lr_scheduler
import os
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks."""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth  # Small value to avoid division by zero

    def forward(self, preds, targets):
        """Computes Dice Loss.
        Args:
            preds: Predicted segmentation mask (logits before sigmoid activation).
            targets: Ground truth mask (binary).
        Returns:
            Dice loss value.
        """
        preds = torch.sigmoid(preds)  # Apply sigmoid to get probabilities
        intersection = (preds * targets).sum(dim=(2, 3))  # Compute intersection
        union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))  # Compute union
        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)  # Dice Score
        dice_loss = 1 - dice_coeff.mean()  # Convert to loss
        return dice_loss

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import numpy as np
from torch.optim import lr_scheduler

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1 variant)"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, prediction, target):
        return torch.mean(torch.sqrt((prediction - target) ** 2 + self.eps ** 2))

class Upsample(nn.Sequential):
    """Upsample module."""
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


import torch
import torch.nn as nn
import numpy as np
import json
from torch.optim import lr_scheduler

import torch
import torch.nn as nn
import numpy as np
import json
from torch.optim import lr_scheduler


# In[7]:


def train_model(net, train_loader, valid_loader, seg_criterion, rec_criterion, optimizer, num_epochs=50, device='cuda', save_path="best_model_recon_dualenc_newnew.pth", results_path="results_full.json"):
    net.to(device)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    best_val_loss = float("inf")  # Track best validation loss for segmentation 2
    area_criterion = nn.MSELoss()  # Define Mean Squared Error loss for area prediction

    for epoch in range(num_epochs):
        net.train()
        epoch_loss = 0.0

        for images, masks, areas, gland_images, gland_masks, gland_areas in train_loader:
            images, masks, areas = images.to(device), masks.to(device), areas.to(device)
            gland_images, gland_masks, gland_areas = gland_images.to(device), gland_masks.to(device), gland_areas.to(device)
            optimizer.zero_grad()
            # Forward pass for nodules
            gland_seg_nodule, nodule_seg, rec_outputs_nodule, area_output_nodule = net(images)
            # Forward pass for glands
            gland_seg_gland, nodule_seg_gland, rec_outputs_gland, area_output_gland = net(gland_images)
            # Compute segmentation and reconstruction losses
            seg_loss1 = seg_criterion(gland_seg_nodule, gland_masks)  # Segmentation loss for gland masks
            seg_loss2 = seg_criterion(nodule_seg, masks)  # Segmentation loss for nodule masks
            rec_loss_nodule = rec_criterion(rec_outputs_nodule, images)  # Reconstruction loss for nodules
            rec_loss_gland = rec_criterion(rec_outputs_gland, gland_images)  # Reconstruction loss for glands
            area_loss = area_criterion(area_output_nodule, areas)  # Area loss for nodules
            
            if epoch < 200:
                loss = 0.5 * rec_loss_nodule + 0.5 * rec_loss_gland  # Phase 1: Use both nodule & gland reconstruction
            elif 200<= epoch < 300:
                loss = 0.8 * seg_loss1 + 0.2 * rec_loss_gland  # Phase 2: Segmentation 1 and reconstruction
            else:
                loss = 0.8 * seg_loss2 + 0.1 * rec_loss_nodule + 0.1 * area_loss  # Phase 3: Segmentation 2 and area
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()


        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # Validation Step
        net.eval()
        val_loss = 0.0
        dice_scores = []
        with torch.no_grad():
            for images, masks,areas, gland_images, gland_masks, gland_areas in valid_loader:
                images, masks, areas = images.to(device), masks.to(device), areas.to(device)
                seg_outputs1, seg_outputs2, rec_outputs, area_output = net(images)
                
                #seg_loss1 = seg_criterion(seg_outputs1, masks)
                seg_loss2 = seg_criterion(seg_outputs2, masks)
                #rec_loss = rec_criterion(rec_outputs, images)
                #area_loss = area_criterion(area_output, areas)
                
                loss = seg_loss2  # Phase 3 validation
                val_loss += loss.item()
    
                preds2 = torch.sigmoid(seg_outputs2)
                preds2 = (preds2 > 0.5).float()
                intersection2 = (preds2 * masks).sum(dim=(2, 3))
                union2 = preds2.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
                dice_score2 = (2. * intersection2 + 1e-6) / (union2 + 1e-6)
                dice_scores.append(dice_score2.mean().item())
    
        avg_val_loss = val_loss / len(valid_loader)
        avg_dice_score = np.mean(dice_scores)
        print(f"Validation Loss: {avg_val_loss:.6f}, Dice Score: {avg_dice_score:.6f}")
    
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(net.state_dict(), save_path)
            print(f"✅ Best model saved with Validation Loss: {best_val_loss:.6f}, Dice Score: {avg_dice_score:.6f} ✅")

    # print("🏆 Training Complete!")
    print("\n🔄 Loading the best model for final evaluation...")
    net.load_state_dict(torch.load(save_path))
    net.to(device)
    net.eval()

    final_dice_scores = []
    with torch.no_grad():
        for batch_idx, (images, masks, areas,_,_,_) in enumerate(valid_loader):
            images, masks, areas = images.to(device), masks.to(device), areas.to(device)
            outputs = net(images)

            # Compute Dice Score
            preds = torch.sigmoid(outputs[1])
            preds = (preds > 0.5).float()

            intersection = (preds * masks).sum(dim=(2, 3))
            union = preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
            dice_score = (2. * intersection + 1e-6) / (union + 1e-6)
            final_dice_scores.append(dice_score.mean().item())

            # if (batch_idx + 1) % 5 == 0 or batch_idx == len(valid_loader) - 1:
            #     print(f"  🎯 Final Evaluation Batch {batch_idx + 1}/{len(valid_loader)} | Dice Score: {dice_scores[-1]:.6f}")

    final_dice = np.mean(final_dice_scores)
    print(f"\n🎯 Final Dice Score on Validation Set: {final_dice:.6f}")

    # **Save Results to JSON**
    results = {
        "Best Dice Score": best_dice_score,
        "Final Dice Score": final_dice
    }
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n📁 Final results saved in `{results_path}` ✅")
    print("\n🏆 🚀 Training Complete! 🎉")


import torch.optim as optim

# vit_name = 'R50-ViT-B_16'
# img_size = 224
# num_classes = 1
# n_skip = 3
# vit_patches_size = 16

# # Initialize the model
# net = get_dual_segmentation_network(
#     vit_name=vit_name, 
#     img_size=img_size, 
#     num_classes=num_classes, 
#     n_skip=n_skip, 
#     vit_patches_size=vit_patches_size
# )

# # Define loss functions
# seg_criterion = DiceLoss()
# rec_criterion = CharbonnierLoss()
# area_criterion = nn.MSELoss()  # MSE for area prediction

# # Optimizer
# optimizer = optim.Adam(net.parameters(), lr=0.001)

# # Train the model
# train_model(
#     net, 
#     train_loader, 
#     test_loader, 
#     seg_criterion, 
#     rec_criterion, 
#     optimizer, 
#     num_epochs=450
# )


# In[ ]:




