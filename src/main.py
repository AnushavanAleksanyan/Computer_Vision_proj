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

print(torch.__version__)
print(torchvision.__version__)

transf = {"train":transforms.Compose([
    transforms.Resize((128,128)),
    #transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485,0.456,0.406],
                        std = [0.229, 0.224, 0.225])]),
    "test":transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485,0.456,0.406],
                        std = [0.229, 0.224, 0.225])
])}

train_data_path = "data/train/"
test_data_path = "data/test/"

train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=transf["train"])
test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=transf["test"])

print("Num Images in Train Dataset:", len(train_data))
print("Num Images in Test Dataset:", len(test_data))

batch_size=16
train_data_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data_loader = data.DataLoader(test_data, batch_size=batch_size)

batch = next(iter(train_data_loader))
images, labels = batch

grid = torchvision.utils.make_grid(images, nrow=10)
plt.figure(figsize=(15,15))
plt.imshow(np.transpose(grid,(1,2,0)))
print("Labels:", labels)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=12*29*29, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=50)
        self.fc3 = nn.Linear(in_features=50, out_features=16)
#         self.act3 = nn.Softmax(dim=1)
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
        # x = F.softmax(x,dim=1)
        return x

network = Net()

sample = next(iter(train_data))
image,label = sample
print(label)
print(image.shape)

pred = network(image.unsqueeze(0))
print(pred.shape)

pred.argmax(dim=1)
batch = next(iter(train_data_loader))
images,labels = batch

preds = network(images)

loss = F.cross_entropy(preds,labels)
loss.item()

loss.backward()
optimizer = optim.Adam(network.parameters(), lr=0.001)

def get_num_correct(preds,labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


for epoch in range(1,5):
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


model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features

model.fc = nn.lineear(num_ftrs, 16)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# scheduler

step_lr_scheduler =lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
