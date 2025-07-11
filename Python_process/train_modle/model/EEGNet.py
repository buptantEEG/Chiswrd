import torch
import torch.nn as nn
import torch.nn.functional as F



class EEGNet(nn.Module): 
    def __init__(self, fs, channel, num_class, signal_length):
        super().__init__()
        fs= fs                  #sampling frequency
        channel= channel              #number of electrode
        num_class= num_class         #number of classes 
        signal_length = signal_length      #number of sample in each tarial
        
        num_input= 1             #number of channel picture (for EEG signal is always : 1)

        F1= 8                    #number of temporal filters
        D= 3                     #depth multiplier (number of spatial filters)
        F2= D*F1                 #number of pointwise filters
        kernel_size_1= (1,round(fs/2)) 
        kernel_size_2= (channel, 1)
        kernel_size_3= (1, round(fs/8))
        kernel_size_4= (1, 1)

        kernel_avgpool_1= (1,4)
        kernel_avgpool_2= (1,8)
        dropout_rate= 0.2

        ks0= int(round((kernel_size_1[0]-1)/2))
        ks1= int(round((kernel_size_1[1]-1)/2))
        kernel_padding_1= (ks0, ks1-1)
        ks0= int(round((kernel_size_3[0]-1)/2))
        ks1= int(round((kernel_size_3[1]-1)/2))
        kernel_padding_3= (ks0, ks1)
        # layer 1
        self.conv2d = nn.Conv2d(num_input, F1, kernel_size_1, padding=kernel_padding_1)
        self.Batch_normalization_1 = nn.BatchNorm2d(F1)
        # layer 2
        self.Depthwise_conv2D = nn.Conv2d(F1, D*F1, kernel_size_2, groups= F1)
        self.Batch_normalization_2 = nn.BatchNorm2d(D*F1)
        self.Elu = nn.ELU()
        self.Average_pooling2D_1 = nn.AvgPool2d(kernel_avgpool_1)
        self.Dropout = nn.Dropout2d(dropout_rate)
        # layer 3
        self.Separable_conv2D_depth = nn.Conv2d( D*F1, D*F1, kernel_size_3,
                                                padding=kernel_padding_3, groups= D*F1)
        self.Separable_conv2D_point = nn.Conv2d(D*F1, F2, kernel_size_4)
        self.Batch_normalization_3 = nn.BatchNorm2d(F2)
        self.Average_pooling2D_2 = nn.AvgPool2d(kernel_avgpool_2)
        # layer 4
        self.Flatten = nn.Flatten()
        self.Dense = nn.Linear(F2*round(signal_length/32), num_class)
        # self.Dense = nn.Linear(1872, num_class)
        
        self.Softmax = nn.Softmax(dim= 1)
        
        
    def forward(self, x):
        # layer 1
        y = self.Batch_normalization_1(self.conv2d(x)) #.relu()
        # layer 2
        y = self.Batch_normalization_2(self.Depthwise_conv2D(y))
        y = self.Elu(y)
        y = self.Dropout(self.Average_pooling2D_1(y))
        # layer 3
        y = self.Separable_conv2D_depth(y)
        y = self.Batch_normalization_3(self.Separable_conv2D_point(y))
        y = self.Elu(y)
        y = self.Dropout(self.Average_pooling2D_2(y))
        # layer 4
        y = self.Flatten(y)
        y = self.Dense(y)
        y = self.Softmax(y)
        return y
        # return F.softmax(y, dim=1), y  # return y for visualization