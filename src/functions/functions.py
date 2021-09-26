# import moduls
import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm


# Data augmentation for train/test data + conversion to tensor
def transform_data():
	transf = {"train":transforms.Compose([
	    transforms.Resize((224,224)),
	    #transforms.CenterCrop(64),
	    transforms.RandomHorizontalFlip(),
	    transforms.RandomRotation(45),
	    transforms.ToTensor(),
	    transforms.Normalize(mean = [0.485,0.456,0.406],
	                        std = [0.229, 0.224, 0.225])]),
	    "test":transforms.Compose([
	    transforms.Resize((224,224)),
	    transforms.ToTensor(),
	    transforms.Normalize(mean = [0.485,0.456,0.406],
	                        std = [0.229, 0.224, 0.225])
	])}
	return transf


def check_size_features_and_labels(data_loaders, dataset):
    train_iter = iter(data_loaders)
    features, labels = next(train_iter)
    print(features.shape,'\n',labels.shape, '\n')
    n_classes = len(dataset.classes)
    print(f'There are {n_classes} different classes.')


# transfer learning
def freeze_until(model, layer_name):
    requires_grad = False
    for name, params in model.named_parameters():
        if layer_name in name:
            requires_grad = True
        params.requires_grad = requires_grad

'''
def train():
	N_EPOCHS = 10
	for epoch in range(N_EPOCHS):
		epoch_loss = 0.0
		for inputs, labels in trainloader:
			inputs = inputs.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			outputs = model(inputs)  # Perform forward pass.
			loss = criterion(outputs, labels)  # Compute loss.
			loss.backward()  # Perform backpropagation; compute gradients.
			optimizer.step()  # Adjust parameters based on gradients.
			epoch_loss += loss.item()  # Accumulate batch loss so we can average over the epoch.
		print("Epoch: {} Loss: {}".format(epoch,
		epoch_loss/len(trainloader)))
'''
def train(n_epochs, model, train_data_loader, device, optimizer, criterion, test_data_loader):
	print_every = 10
	valid_loss_min = np.Inf
	val_loss = []
	val_acc = []
	train_loss = []
	train_acc = []
	total_step = len(train_data_loader)
	for epoch in tqdm(range(1, n_epochs+1)):
	    model.train()
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
	        running_train_acc = float(correct)/float(data_.shape[0])
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



