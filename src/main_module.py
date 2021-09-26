import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import transforms
from torchvision import models
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import model_argparse
from functions.functions import transform_data, check_size_features_and_labels, train, freeze_until

# version check
print(torch.__version__)
print(torchvision.__version__)


def main(epochs, model, b_size):
    # define dataset
    train_data_path = "data/train/"
    test_data_path = "data/test/"
    transf = transform_data()

    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=transf["train"])
    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=transf["test"])

    print("Num Images in Train Dataset:", len(train_data))
    print("Num Images in Test Dataset:", len(test_data))


    # Data loading
    batch_size=b_size
    train_data_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data_loader = data.DataLoader(test_data, batch_size=batch_size)


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("loaded data size", len(train_data_loader))
    batch = next(iter(train_data_loader))
    images, labels = batch

    grid = torchvision.utils.make_grid(images, nrow=10)
    plt.figure(figsize=(15,15))
    plt.imshow(np.transpose(grid,(1,2,0)))
    print("Labels:", labels)

    # class Net(nn.Module):
    #     def __init__(self):
    #         super(Net, self).__init__()
    #         self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
    #         self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
            
    #         self.fc1 = nn.Linear(in_features=12*29*29, out_features=84)
    #         self.fc2 = nn.Linear(in_features=84, out_features=50)
    #         self.fc3 = nn.Linear(in_features=50, out_features=16)
    #         #self.act3 = nn.Softmax(dim=1)
    #     def forward(self, x):
    #         # (1) input layer
    #         x = x
            
    #         # (2) hidden conv layer
    #         x = self.conv1(x)
    #         x = F.relu(x)
    #         x = F.max_pool2d(x, kernel_size=2, stride=2)
            
    #         # (3) hidden conv layer
    #         x = self.conv2(x)
    #         x = F.relu(x)
    #         x = F.max_pool2d(x, kernel_size=2, stride=2)
            
    #         # (4) hidden linear layer
    #         x = x.reshape(-1, 12*29*29)
    #         x = self.fc1(x)
    #         x = F.relu(x)

    #         # (5) hidden linear layer
    #         x = self.fc2(x)
    #         x = F.relu(x)
            
    #         # (6) output layer
    #         x = self.fc3(x)
    #         # x = F.softmax(x,dim=1)
    #         return x



    # transfer learning
    if model == "r18":
        model = models.resnet18(pretrained=True)
    elif model =="r34":
        model = models.resnet34(pretrained=True)
    elif model =="r50":
        model = models.resnet50(pretrained=True)
    else:
        print("Choose another model")

    freeze_until(model, "fc")
    # for param  in model.parameters():
    #     param.requires_grad = False


    def accuracy(out, labels):
        _,pred = torch.max(out, dim=1)
        return torch.sum(pred==labels).item()

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 16)

    model.to(device)


    check_size_features_and_labels(train_data_loader, train_data)

    # run from terminal
    # parser = argparse.ArgumentParser(description='Image classification model')
    # parser.add_argument('-ep','--epochs', metavar="", type=int, required=True, help='Number of epochs')
    # args = parser.parse_args()

    # Training
    n_epochs = epochs
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    train(n_epochs, model, train_data_loader, device, optimizer, criterion, test_data_loader)


if __name__ == '__main__':
    print('Please run this code from terminal (cmd)')