## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout(p=0.1)

        self.conv2 = nn.Conv2d(32,64,3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.drop2 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv2d(64,128,3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.drop3 = nn.Dropout(p=0.3)

        self.conv4 = nn.Conv2d(128,256,3)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.bn4 = nn.BatchNorm2d(256)
        self.drop4 = nn.Dropout(p=0.4)

        self.conv5 = nn.Conv2d(256,512,1)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.bn5 = nn.BatchNorm2d(512)
        self.drop5 = nn.Dropout(p=0.5)
        

        #Linear Layer
        self.fc1 = nn.Linear(512*6*6, 1024)
        self.drop6 = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(1024, 136)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool1(F.elu(self.bn1(self.conv1(x))))
        x = self.drop1(x)
        x = self.pool2(F.elu(self.bn2(self.conv2(x))))
        x = self.drop2(x)
        x = self.pool3(F.elu(self.bn3(self.conv3(x))))
        x = self.drop3(x)
        x = self.pool4(F.elu(self.bn4(self.conv4(x))))
        x = self.drop4(x)
        x = self.pool5(F.elu(self.bn5(self.conv5(x))))
        x = self.drop5(x)
        x = x.view(x.size(0), -1)
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
