import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self, nb_classes=4, Chans=22, Samples=401,
                 dropoutRate=0.5, kernLength=64, F1=16,
                 D=2, F2=32, norm_rate=0.25):
        super(EEGNet, self).__init__()

        self.chans = Chans
        self.samples = Samples
        
       

        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.dw_conv1 = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.activation = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 =  nn.Dropout(dropoutRate)

        self.sep_conv1 = nn.Conv2d(F1 * D, F1 * D, (1, 16), groups=F1 * D, padding=(0, 8), bias=False)
        self.conv2 = nn.Conv2d(F1 * D, F2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 =  nn.Dropout(dropoutRate)

        self.flatten = nn.Flatten()
        self.dense = nn.Linear(F2 * (Samples // 32), nb_classes)

      

    def forward(self, x):
        x = x.view(-1,  1, self.chans, self.samples)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dw_conv1(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)

        x = self.sep_conv1(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.dense(x)
      

        return x
