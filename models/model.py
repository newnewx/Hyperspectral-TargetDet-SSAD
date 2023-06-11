import torch
from torch import nn
from torch.nn import functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Block1(nn.Module):
    def __init__(self, input_nc):
        super(Block1, self).__init__()
        self.conv1 = nn.Conv2d(input_nc, input_nc, kernel_size=3, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(input_nc)
        
        self.conv2 = nn.Conv2d(input_nc, input_nc, kernel_size=3, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(input_nc)
        
        self.ca = ChannelAttention(input_nc)
        self.sa = SpatialAttention()
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x   

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.ca(out) * out
        out = self.sa(out) * out

        out +=residual

        out = self.relu(out)
        return out  

class Block2(nn.Module):
    def __init__(self, input_nc):
        super(Block2, self).__init__()
        self.conv1 = nn.Conv2d(input_nc, 2*input_nc, kernel_size=3, bias=False, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(2*input_nc)
        
        self.conv2 = nn.Conv2d(2*input_nc, 2*input_nc, kernel_size=3, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(2*input_nc)
        
        self.ca = ChannelAttention(2*input_nc)
        self.sa = SpatialAttention()
        
        self.conv3 = nn.Conv2d(input_nc, 2*input_nc, kernel_size=1, bias=False, stride=2)
        self.bn3 = nn.BatchNorm2d(2*input_nc)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.conv3(x)
        residual = self.bn3(residual)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.ca(out) * out
        out = self.sa(out) * out

        out +=residual

        out = self.relu(out)
        return out

class Block3(nn.Module):
    def __init__(self, input_nc):
        super(Block3, self).__init__()
        # (n-1)*stride-2*p+k+outputpadding
        self.conv1 = nn.ConvTranspose2d(input_nc, int(1/2*input_nc), kernel_size=3, bias=False, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(int(1/2*input_nc))
        
        self.conv2 = nn.Conv2d(int(1/2*input_nc), int(1/2*input_nc), kernel_size=3, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(int(1/2*input_nc))
        
        self.ca = ChannelAttention(int(1/2*input_nc))
        self.sa = SpatialAttention()

        self.conv3 = nn.ConvTranspose2d(input_nc, int(1/2*input_nc), kernel_size=1, bias=False, stride=2, output_padding=1)
        self.bn3 = nn.BatchNorm2d(int(1/2*input_nc))

        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = self.conv3(x)
        residual = self.bn3(residual)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.ca(out) * out
        out = self.sa(out) * out

        out +=residual

        out = self.relu(out)
        return out

class network(nn.Module):

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)

    def __init__(self, input_nc, output_nc):
        super(network, self).__init__()
        # Encoder
        self.EL01 = nn.Conv2d(input_nc, 64, kernel_size=3, padding=1)
        self.EL02 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.EL11 = Block1(64)
        self.EL12 = Block1(64)
        
        self.EL21 = Block2(64)
        self.EL22 = Block1(128)

        self.EL31 = Block2(128)
        self.EL32 = Block1(256)

        self.EL41 = Block2(256)
        self.EL42 = Block1(512)

        # # Decoder
        self.DL41 = Block1(512)
        self.DL42 = Block3(512)

        self.DL31 = Block1(256)
        self.DL32 = Block3(256)

        self.DL21 = Block1(128)
        self.DL22 = Block3(128)

        self.DL11 = Block1(64)
        self.DL12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.DL13 = nn.BatchNorm2d(64)
        self.relu_out = nn.ReLU(inplace=True)

        self.DL0 = nn.ConvTranspose2d(64, output_nc, kernel_size=3, bias=False, stride=2, padding=1, output_padding=1)
        self.apply(self.weight_init)

        
    def forward(self, x):   
        # Encoder
        out = self.relu(self.EL02(self.EL01(x))) 
        out = F.max_pool2d(out, kernel_size=2, stride=2) 

        out = self.EL12(self.EL11(out)) 

        out = self.EL22(self.EL21(out)) 

        out = self.EL32(self.EL31(out)) 

        out = self.EL42(self.EL41(out)) 

        # # # Decoder

        out = self.DL42(self.DL41(out)) 

        out = self.DL32(self.DL31(out)) 

        out = self.DL22(self.DL21(out)) 
        
        # out = self.DL11(out)
        out = self.DL13(self.DL12(self.DL11(out))) 
        out = self.relu_out(out) 

        # last stage
        out = self.DL0(out) 
        output = torch.sigmoid(out)     

        return output