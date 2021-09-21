# import moduls
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

