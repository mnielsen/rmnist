"""generate_abstract_features.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the images in RMNIST through a truncated version of ResNet-18, and
save the features in the final layer.  Based on the transfer learning
tutorial and code by Sasank Chilamkurthy, at
http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html.
Note that we ignore the test data. A more thorough treatment would
consider validation and test data separately.

"""

# Standard library
from __future__ import print_function, division
import cPickle
import gzip

# My libraries
import data_loader

# Third-party libraries
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms


# Define the truncated model
net = models.resnet18(pretrained=True)
for param in net.parameters():
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

class RMNIST(Dataset):

    def __init__(self, n=0, train=True, transform=None, expanded=False):
        self.n = n
        self.transform = transform
        td, vd, ts = data_loader.load_data(n, expanded=expanded)
        if train: self.data = td
        else: self.data = vd
        
    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        data = self.data[0][idx]
        img = (data*256)
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


# Compute the abstract features for the validation data
data_transform = transforms.Compose(
    [transforms.Scale(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
data_loader_val = torch.utils.data.DataLoader(
    RMNIST(10, train=False, transform=data_transform), batch_size=100)
print("\nComputing features for validation data")
for j in range(100):
    print("Computing features for batch {} of 100".format(j))
    inputs, labels = next(iter(data_loader_val))
    inputs, labels = Variable(inputs), Variable(labels)
    outputs = forward_partial(net, inputs)
    if j == 0:
        vd2 = (outputs.data.numpy(), labels.data.numpy())
    else:
        vd2 = (
            np.concatenate((vd2[0], outputs.data.numpy())),
            np.concatenate((vd2[1], labels.data.numpy()))
        )

# Compute the abstract features for the RMNIST training data
for n in [1, 5, 10]:
    print("\nComputing features for n = {}".format(n))
    data_loader_train = torch.utils.data.DataLoader(
        RMNIST(n, transform=data_transform), batch_size=n*10)
    # Do everything in one batch
    inputs, labels = next(iter(data_loader_train))
    inputs, labels = Variable(inputs), Variable(labels)
    outputs = forward_partial(net, inputs)
    td2 = (outputs.data.numpy(), labels.data.numpy())
    f = gzip.open("data/rmnist_abstract_features_{}.pkl.gz".format(n), 'wb')
    cPickle.dump((td2, vd2, (0,0)), f)
    f.close()






