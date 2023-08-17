"""

This script stores the Neural Networks (fully connected NN, convolutional neural networks) for the training of mineral_classification

"""

import torch
import torch.nn as nn

class NN(nn.Module):
    def __init__(self, input_dim, dropout_fc, number_of_minerals):
        super(NN, self).__init__()
        self.input_layer = nn.Linear(input_dim, 64)
        self.hidden_layer1 = nn.Linear(64, 256)
        self.hidden_layer2 = nn.Linear(256, 1024)
        self.hidden_layer3 = nn.Linear(1024, 1024)
        self.hidden_layer4 = nn.Linear(1024, 64)
        self.output_layer = nn.Linear(64, number_of_minerals)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_fc)
        # self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        out = self.relu(self.input_layer(x))
        out = self.relu(self.hidden_layer1(out))
        out = self.dropout(out) # apply drop-out
        out = self.relu(self.hidden_layer2(out))
        out = self.dropout(out) # apply drop-out
        out = self.relu(self.hidden_layer3(out))
        out = self.dropout(out) # apply drop-out
        out = self.relu(self.hidden_layer4(out))
        # out = self.dropout(out)
        out = self.output_layer(out)
        # print('out.shape before softmax', out.shape)
        # out = self.softmax(out) # gives a problem when turned on!
        # when softmax turned on: [0.0004526386910583824, 9.426586984773166e-06, 0.9988535642623901, 1.103805588598128e-11, 2.390627378190402e-05, 6.600160418201995e-07, 0.0006591785931959748, 6.815116080360895e-07]
        # print('out', out)
        return out

# class CNN_FullSpectrum(nn.Module):
#     def __init__(self, input_dim, dropout_fc, dropout_conv, number_of_minerals):
#         super(CNN_FullSpectrum, self).__init__()
#         self.n_after_conv = 300
#
#         # CNN-LIBS-Large
#         self.conv1 = nn.Conv1d(1, 100, kernel_size=16, stride=8)  # out_channels refers to the number of filters
#         self.conv2 = nn.Conv1d(100, 200, kernel_size=16, stride=8, padding=1)
#         self.conv3 = nn.Conv1d(200, 250, kernel_size=10, stride=5)
#         self.conv4 = nn.Conv1d(250, 275, kernel_size=10, stride=5)
#         self.conv5 = nn.Conv1d(275, self.n_after_conv, kernel_size=9, stride=5)
#
#         self.fc1 = nn.Linear(self.n_after_conv, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, number_of_minerals)
#
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.drop = nn.Dropout(p=dropout_conv)
#
#
#     def forward(self, x):
#         # print(x.shape)
#         # x = x.unsqueeze(2)
#         # x = x.view(1000, 1, 1)
#         print('x.shape at start is', x.shape) # [1, 1, 16376]
#
#         x = self.relu(self.conv1(x))
#         print('x.shape after conv1 is', x.shape) #[1, 100, 2046]
#
#         # x = self.maxpool(x)
#         # print('x.shape after maxpool1 is', x.shape) #[]
#
#         x = self.relu(self.conv2(x))
#         # x = self.maxpool(x)
#         print('x.shape after conv2 is', x.shape) #[1, 200, 255]
#
#         x = self.relu(self.conv3(x))
#         # x = self.maxpool(x)
#         print('x.shape after conv3 is', x.shape) #[1, 250, 50]
#
#         x = self.relu(self.conv4(x))
#         # x = self.maxpool(x)
#         print('x.shape after conv4 is', x.shape) #[1, 275, 9]
#
#         x = self.relu(self.conv5(x))
#         # x = self.maxpool(x)
#         print('x.shape after conv5 is', x.shape) #[1, 300, 1]
#
#         # Flatten
#         x = x.view(-1, self.n_after_conv)
#         # print('x.shape after fc is', x.shape)
#
#         x = self.relu(self.fc1(x))
#         x = self.drop(x)
#         # print('x.shape after fc1 is', x.shape)
#
#         x = self.relu(self.fc2(x))
#         # print('x.shape after fc2 is', x.shape)
#         x = self.drop(x)
#
#         x = self.relu(self.fc3(x))
#         # print('x.shape after fc3 is', x.shape)
#
#         # x = self.softmax(x)
#         return x
#
# class CNN_ROI(nn.Module):
#     def __init__(self, input_dim, dropout_fc, dropout_conv, number_of_minerals):
#         super(CNN_ROI, self).__init__()
#         self.n_after_conv = 300
#
#         # CNN-LIBS-2
#         self.conv1 = nn.Conv1d(1, 100, kernel_size=10, stride=10) # out_channels refers to the number of filters
#         self.conv2 = nn.Conv1d(100, 200, kernel_size=5, stride=5)
#         self.conv3 = nn.Conv1d(200, 250, kernel_size=5, stride=5)
#         self.conv4 = nn.Conv1d(250, self.n_after_conv, kernel_size=4, stride=4)
#
#         self.fc1 = nn.Linear(self.n_after_conv, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, number_of_minerals)
#
#         self.relu = nn.ReLU(inplace=True)
#
#         self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
#
#         self.drop_fc = nn.Dropout(p=dropout_fc)
#         self.drop_conv = nn.Dropout(p=dropout_conv)
#
#         self.avgpool = nn.AdaptiveAvgPool1d(6)
#
#         self.softmax = nn.Softmax()
#
#         # Batchnormalization (something Pierre always does)
#         # self.batchnorm = nn.BatchNorm1d(200)
#
#     def forward(self, x):
#         # print(x.shape)
#         # x = x.unsqueeze(2)
#         # x = x.view(1000, 1, 1)
#         # print('x.shape at start is', x.shape) # [1, 1, 2000]
#         # x = self.maxpool(x)
#         # print('x.shape after maxpool1 is', x.shape) #[1, 1, 1000 ]
#
#         x = self.drop_conv(self.relu(self.conv1(x)))
#         # print('x.shape after conv1 is', x.shape) #[1, 100, 100]
#
#
#         x = self.drop_conv(self.relu(self.conv2(x)))
#         # x = self.maxpool(x)
#         # print('x.shape after conv2 is', x.shape) #[1, 200, 20]
#
#         x = self.drop_conv(self.relu(self.conv3(x)))
#         # x = self.maxpool(x)
#         # print('x.shape after conv3 is', x.shape) #[1, 250, 4]
#
#         x = self.drop_conv(self.relu(self.conv4(x)))
#         # x = self.maxpool(x)
#         # print('x.shape after conv4 is', x.shape) #[1, 300, 1]
#
#         # Flatten
#         x = x.view(-1, self.n_after_conv)
#         # print('x.shape after fc is', x.shape)
#
#         x = self.relu(self.fc1(x))
#         x = self.drop_fc(x)
#         # print('x.shape after fc1 is', x.shape)
#
#         x = self.relu(self.fc2(x))
#         # print('x.shape after fc2 is', x.shape)
#         x = self.drop_fc(x)
#
#         x = self.relu(self.fc3(x))
#         # print('x.shape after fc3 is', x.shape)
#
#         # x = self.softmax(x)
#         return x