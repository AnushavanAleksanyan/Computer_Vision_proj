import torchvision
from torchvision import transforms
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchvision import models



resnet = models.resnet18(pretrained=True)

print(resnet)