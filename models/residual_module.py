import torch.nn as nn

class Residual(nn.Module):

    def __init__(self,input_channel_num,ouput_channnel_num):
        super(Residual,self).__init__()
        self.input_channel_num=input_channel_num
        self.ouput_channnel_num=ouput_channnel_num
        self.bn1=nn.BatchNorm2d(self.input_channel_num)
        self.relu=nn.ReLU(inplace=True)
        self.conv1=nn.Conv2d(self.input_channel_num,self.ouput_channnel_num//2,bias=True,kernel_size=1,stride=1)
        self.bn2=nn.BatchNorm2d(self.ouput_channnel_num//2)
        self.conv2=nn.Conv2d(self.ouput_channnel_num//2,self.ouput_channnel_num//2,bias=True,kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm2d(self.ouput_channnel_num//2)
        self.conv3=nn.Conv2d(self.ouput_channnel_num//2,self.ouput_channnel_num,bias=True,kernel_size=1)

        if self.input_channel_num!=self.ouput_channnel_num:
            self.conv4=nn.Conv2d(self.input_channel_num,self.ouput_channnel_num,bias=True,kernel_size=1)

    def forward(self, x):
        residual=x
        out=self.bn1(x)
        out=self.relu(out)
        out=self.conv1(out)
        out=self.bn2(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn3(out)
        out=self.relu(out)
        out=self.conv3(out)

        if self.input_channel_num!=self.ouput_channnel_num:
            residual=self.conv4(x)

        return out+residual
