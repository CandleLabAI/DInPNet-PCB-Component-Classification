# ------------------------------------------------------------------------
# Copyright (c) 2023 CandleLabAI. All Rights Reserved.
# ------------------------------------------------------------------------

"""
DInPNet model architecture
"""
from involution import Involution2d
from torch import nn
import torch

class IBR(nn.Module):
    """
        This class defines the Involution layer with batch normalization and PReLU activation
    """
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        Args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = Involution2d(in_channels=nIn, out_channels=nOut, kernel_size=(kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.batchnorm = nn.BatchNorm2d(nOut, eps=1e-03)
        self.activation = nn.PReLU(nOut)

    def forward(self, data):
        """
        Args:
            data: input feature map
        return:
            transformed feature map
        """
        output = self.conv(data)
        output = self.batchnorm(output)
        output = self.activation(output)
        return output

class IBRDilated(nn.Module):
    """
    This class defines the dilated involution.
    """
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        Args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: stride rate for down-sampling. Default is 1
            d: optional dilation rate
        """
        super().__init__()
        padding = int((kSize - 1)/2) * d
        self.conv = Involution2d(in_channels=nIn, out_channels=nOut, kernel_size=(kSize, kSize), stride=stride, padding=(padding, padding), bias=False, dilation=d)
        self.batchnorm = nn.BatchNorm2d(nOut, eps=1e-03)
        self.activation = nn.PReLU(nOut)

    def forward(self, data):
        """
        Args:
            data: input feature map
        return: 
            transformed feature map
        """
        output = self.conv(data)
        output = self.batchnorm(output)
        output = self.activation(output)
        return output

class DInPBlock(nn.Module):
    """
    This class defines the DInPBlock
    """
    def __init__(self, nIn, nOut):
        """
        Args:
            nIn: number of input channels
            nOut: number of output channels
        """
        super().__init__()
        n = int(nOut/5)
        n1 = nOut - 4*n
        self.invo1 = IBR(nIn, n, 3, 2)
        self.dilated_invo1 = IBRDilated(n, n1, 3, 1, 1)
        self.dilated_invo2 = IBRDilated(n, n, 3, 1, 2)
        self.dilated_invo4 = IBRDilated(n, n, 3, 1, 4)
        self.dilated_invo8 = IBRDilated(n, n, 3, 1, 8)
        self.dilated_invo16 = IBRDilated(n, n, 3, 1, 16)
        self.batchnorm = nn.BatchNorm2d(nOut, eps=1e-3)
        self.activation = nn.PReLU(nOut)

    def forward(self, data):
        """
        Args:
            data: input feature map
        return: 
            transformed feature map
        """
        output1 = self.invo1(data)
        dilated_invo1 = self.dilated_invo1(output1)
        dilated_invo2 = self.dilated_invo2(output1)
        dilated_invo4 = self.dilated_invo4(output1)
        dilated_invo8 = self.dilated_invo8(output1)
        dilated_invo16 = self.dilated_invo16(output1)

        add1 = dilated_invo2
        add2 = add1 + dilated_invo4
        add3 = add2 + dilated_invo8
        add4 = add3 + dilated_invo16

        combine = torch.cat([dilated_invo1, add1, add2, add3, add4], 1)
        output = self.batchnorm(combine)
        output = self.activation(output)
        return output

class DInPNet(nn.Module):
    """
    This class defines the proposed DInPNet network
    """
    def __init__(self, num_classes = 6):
        """
        Args:
            num_classes: number of classes for classification
        """
        super().__init__()
        self.invo1 = Involution2d(3, 16, kernel_size = 3, stride = 1, padding = 1)
        self.invo2 = Involution2d(32, 64, kernel_size = 3, stride = 1, padding = 1)
        self.batchnorm1 = nn.BatchNorm2d(16, eps=1e-3)
        self.batchnorm2 = nn.BatchNorm2d(64, eps=1e-3)
        self.activation = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.dinpblock = DInPBlock(16, 32)
        self.linear_1 = nn.Linear(64*8*8, 128)
        self.linear_2 = nn.Linear(128, num_classes)

    def forward(self, data):
        """
        Args:
            data: input feature map
        return: 
            transformed feature map
        """
        output = self.invo1(data)
        output = self.batchnorm1(output)
        output = self.activation(output)
        output = self.maxpool(output)
        output = self.dinpblock(output)
        output = self.maxpool(output)
        output = self.invo2(output)
        output = self.batchnorm2(output)
        output = self.activation(output)
        output = output.view(output.size(0), 64*8*8)
        output = self.linear_1(output)
        output = self.activation(output)
        output = self.linear_2(output)
        return output

def get_model(**kwargs):
    """
    Create model definition
    """
    model = DInPNet(**kwargs)
    return model
