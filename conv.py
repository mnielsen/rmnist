from __future__ import print_function

import data_loader

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset

# Training settings
n = 1 # use RMNIST/n
batch_size = 10
if n > 0: epochs = 2000/n
else: epochs = 40
if n > 0: lr = 0.1
else: lr = 0.01
momentum = 0.0
mean_data_init = 0.1
sd_data_init = 0.25
seed = 1
expanded=True

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean_data_init,), (sd_data_init,))
])
torch.manual_seed(seed)

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
        img = Image.fromarray(np.uint8(img))
        if self.transform:
            img = self.transform(img)
        value = self.data[1][idx]
        return (img, value)

train_dataset = RMNIST(n, train=True, transform=transform, expanded=expanded)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
training_data = list(train_loader)

validation_dataset = RMNIST(n, train=False, transform=transform, expanded=expanded)
validation_loader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=1000, shuffle=True)
validation_data = list(validation_loader)

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

def train(epoch):
    global lr, optimizer
    model.train()
    for batch_idx, (data, target) in enumerate(training_data):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print('Training epoch: {}'.format(epoch))
        lr *= 0.8
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

def accuracy():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in validation_data:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        validation_loss += F.nll_loss(output, target, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    validation_loss /= 10000
    if epoch % 10 == 0:
        print('Validation set: average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)\n'.format(
            validation_loss, correct, 10000, 100. * correct / 10000))


for epoch in range(1, epochs + 1):
    train(epoch)
    accuracy()
