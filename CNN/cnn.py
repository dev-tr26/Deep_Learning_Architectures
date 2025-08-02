import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets



# simple fully commected NN

class NN(nn.Module):
    def __init__(self,input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class CNN(nn.Module):
    def __init__(self,in_channels=1,num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels=8,kernel_size=(3,3),padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv1 = nn.Conv2d(in_channels,out_channels=16,kernel_size=(3,3),padding=(1,1))
        self.fc1 = nn.Linear(16*7*7, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        
        return x
    

model = CNN()
x = torch.random.randn(64, 1, 28, 28) 
       
    
    
    
    
    
    
    
    
    
    
    
    
    
        