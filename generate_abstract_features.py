from __future__ import print_function, division

import cPickle
import gzip

import data_loader

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os

data_transforms = {
    'train': transforms.Compose([
        transforms.Scale(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class RMNIST(Dataset):

    def __init__(self, n=0, train=True, transform=None, expanded=False):
        self.n = n
        self.transform = transform
        td, vd, ts = data_loader.load_data(n, expanded=expanded)
        if train: self.data = td
        else: self.data = (vd[0], vd[1])
        
    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        data = self.data[0][idx]
        img = (data*256) # 255-(data*256) gives us a white background
        img = img.reshape(28, 28)
        imgColor = np.zeros((28, 28, 3), 'uint8')
        imgColor[..., 0] = img
        imgColor[..., 1] = img
        imgColor[..., 2] = img
        imgColor = Image.fromarray(imgColor, mode="RGB")
        if self.transform:
            imgColor = self.transform(imgColor)
        value = self.data[1][idx]
        return (imgColor, value)





model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

def forward_partial(model, x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    
    x = model.avgpool(x)
    x = x.view(x.size(0), -1)
    return x


dataloders = {"val": torch.utils.data.DataLoader(
    RMNIST(10, train=False, transform=data_transforms["val"]), batch_size=100)
}
for j in range(100):
    print(j)
    inputs, labels = next(iter(dataloders["val"]))
    inputs, labels = Variable(inputs), Variable(labels)
    outputs = forward_partial(model_conv, inputs)
    if j == 0:
        vd2 = (outputs.data.numpy(), labels.data.numpy())
    else:
        vd2 = (
            np.concatenate((vd2[0], outputs.data.numpy())),
            np.concatenate((vd2[1], labels.data.numpy()))
        )

for n in [1, 5, 10]:
    print("n = {}".format(n))
    dataloders["train"] = torch.utils.data.DataLoader(
        RMNIST(n, transform=data_transforms["train"]), batch_size=n*10)
    inputs, labels = next(iter(dataloders["train"]))
    inputs, labels = Variable(inputs), Variable(labels)
    outputs = forward_partial(model_conv, inputs)
    td2 = (outputs.data.numpy(), labels.data.numpy())
    f = gzip.open("data/rmnist_abstract_features_{}.pkl.gz".format(n), 'wb')
    cPickle.dump((td2, vd2, (0,0)), f)
    f.close()






