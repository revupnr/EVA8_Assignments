from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# CUDA?

print(torch.__version__)

cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

dropout_value = 0.1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.init_batch_norm()
        self.init_ultimus()
        self.dropout = nn.Dropout(dropout_value)

    def init_ultimus(self):
        self.fcK = nn.Linear(in_features=48,out_features=8,bias=False)
        self.fcQ = nn.Linear(in_features=48,out_features=8,bias=False)
        self.fcV = nn.Linear(in_features=48,out_features=8,bias=False)

    def ultimus(self,x):
        K = self.fcK(x)
        Q = self.fcQ(x)
        V = self.fcV(x)

        # AM = SoftMax(QTK)/(8^0.5) = 8*8 = 8
        QTK = torch.cross(torch.transpose(Q),K)
        AM = F.softmax(QTK)/(8^(0.5))

        # Z = V*AM = 8*8 > 8
        Z = torch.cross(V,AM)
        fc_out = nn.Linear(in_features=8,out_features=48,bias=False)
        return fc_out(Z)

    def init_batch_norm(self):
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False,dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) 

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False,dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) 

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(3, 3), padding=1, bias=False,dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(dropout_value)
        ) 
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=32)
        ) 

        # Then add final FC layer that converts 48 to 10 and sends it to the loss function
        self.fc1 = nn.Linear(in_features=48,out_features=10,bias=False)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)

        x = self.gap(x) 

        x = self.ultimus(x) 
        x = self.ultimus(x) 
        x = self.ultimus(x) 
        x = self.ultimus(x) 
        
        x = self.fc1(x)
        x = x.view(-1, 10) 
        F.log_softmax(x, dim=-1)
        return x