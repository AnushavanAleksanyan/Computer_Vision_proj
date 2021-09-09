import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import transforms
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import freeze_support
from multiprocessing import Pool


print("Torch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)

def get_num_correct(preds,labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=12*29*29, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=50)
        self.fc3 = nn.Linear(in_features=50, out_features=16)
    def forward(self, x):
        # (1) input layer
        x = x
        
        # (2) hidden conv layer
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # (3) hidden conv layer
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # (4) hidden linear layer
        x = x.reshape(-1, 12*29*29)
        x = self.fc1(x)
        x = F.relu(x)

        # (5) hidden linear layer
        x = self.fc2(x)
        x = F.relu(x)
        
        # (6) output layer
        x = self.fc3(x)
        # x = F.softmax(x, dim=1)
        return x


transforms = transforms.Compose([
    transforms.Resize((128,128)),
    #transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485,0.456,0.406],
                        std = [0.229, 0.224, 0.225])
])


network = Net()

train_data_path = "data/train/"
#test_data_path = "data/test/"


train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=transforms)
# test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=transforms)

print("\nNum Images in Train Dataset:", len(train_data))
#print("Num Images in Test Dataset:", len(test_data))
if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    batch_size=16
    train_data_loader = data.DataLoader(train_data, batch_size=batch_size,num_workers=4)

    optimizer = optim.Adam(network.parameters(), lr=0.001)

    for epoch in range(5):
        total_loss = 0
        total_correct = 0
        for batch in train_data_loader:
            images,labels = batch

            preds = network(images)
            loss = F.cross_entropy(preds,labels) # Calculate Loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss+=loss.item()
            total_correct+=get_num_correct(preds, labels)
        print("epoch:", epoch, "total_correct:", total_correct, "loss:", total_loss)



