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

batch_size=32
train_data_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data_loader = data.DataLoader(test_data, batch_size=batch_size)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(len(train_data_loader))
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
        #self.act3 = nn.Softmax(dim=1)
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

# def train_model(model):
#     for epoch in range(1,5):
#         total_loss = 0
#         total_correct = 0
#         for batch in train_data_loader:
#             images,labels = batch

#             preds = network(images)
#             loss = F.cross_entropy(preds,labels) # Calculate Loss

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss+=loss.item()
#             total_correct+=get_num_correct(preds, labels)
#         print("epoch:", epoch, "total_correct:", total_correct, "loss:", total_loss)


model = models.resnet18(pretrained=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

def accuracy(out, labels):
    _,pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 16)
model.to(device)


n_epochs = 1
print_every = 10
valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(train_data_loader)
for epoch in range(1, n_epochs+1):
    running_loss = 0.0
    correct = 0
    total=0
    print(f'Epoch {epoch}\n')
    for batch_idx, (data_, target_) in enumerate(train_data_loader):
        data_, target_ = data_.to(device), target_.to(device)
        optimizer.zero_grad()
        
        outputs = model(data_)
        loss = criterion(outputs, target_)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _,pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred==target_).item()
        total += target_.size(0)
        if (batch_idx) % 20 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
    train_acc.append(100 * correct / total)
    train_loss.append(running_loss/total_step)
    print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')
    batch_loss = 0
    total_t=0
    correct_t=0
    with torch.no_grad():
        model.eval()
        for data_t, target_t in (test_data_loader):
            data_t, target_t = data_t.to(device), target_t.to(device)
            outputs_t = model(data_t)
            loss_t = criterion(outputs_t, target_t)
            batch_loss += loss_t.item()
            _,pred_t = torch.max(outputs_t, dim=1)
            correct_t += torch.sum(pred_t==target_t).item()
            total_t += target_t.size(0)
        val_acc.append(100 * correct_t/total_t)
        val_loss.append(batch_loss/len(test_data_loader))
        network_learned = batch_loss < valid_loss_min
        print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')

        
        if network_learned:
            valid_loss_min = batch_loss
            torch.save(model.state_dict(), 'resnet.pt')
            print('Improvement-Detected, save-model')
    model.train()


# fig = plt.figure(figsize=(20,10))
# plt.title("Train-Validation Accuracy")
# plt.plot(train_acc, label='train')
# plt.plot(val_acc, label='validation')
# plt.xlabel('num_epochs', fontsize=12)
# plt.ylabel('accuracy', fontsize=12)
# plt.legend(loc='best')
# plt.show()

# evaluate the model
# acc = evaluate_model(test_data_loader, model)
# print('Accuracy: %.3f' % acc)

def visualize_model(net, num_images=4):
    images_so_far = 0
    fig = plt.figure(figsize=(15, 10))
    
    for i, data in enumerate(test_data_loader):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        preds = preds.cpu().numpy()
        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(2, num_images//2, images_so_far)
            ax.axis('off')
            ax.set_title('predictes: {}'.format(test_data.classes[preds[j]]))
            plt.imshow(inputs[j])
            
            if images_so_far == num_images:
                return 

plt.ion()
visualize_model(model)
plt.ioff()
plt.show()