import torch.nn as nn
from models.residual_module import Residual

class Hourglass(nn.Module):

    def __init__(self,sampling_num,residual_num,channel_num):
        super(Hourglass,self).__init__()
        self.sampling_num=sampling_num
        self.residual_num=residual_num
        self.channel_num=channel_num

        _upper,_down1_residual,_down2_residual,_down3_residual=[],[],[],[]

        for i in range(self.residual_num):
            _upper.append(Residual(self.channel_num,self.channel_num))

        self.down1=nn.MaxPool2d(kernel_size=2,stride=2)
        for i in range(self.residual_num):
            _down1_residual.append(Residual(self.channel_num,self.channel_num))
        if self.sampling_num>1:
            self.down2=Hourglass(sampling_num-1,self.residual_num,self.channel_num)
        else:
            for i in range(self.residual_num):
                _down2_residual.append(Residual(self.channel_num,self.channel_num))
            self.down2_residual=nn.ModuleList(_down2_residual)

        for i in range(self.residual_num):
            _down3_residual.append(Residual(self.channel_num,self.channel_num))

        self.upper=nn.ModuleList(_upper)
        self.down1_residual=nn.ModuleList(_down1_residual)
        self.down3_residual=nn.ModuleList(_down3_residual)

        self.upsampling=nn.Upsample(scale_factor=2)

    def forward(self, x):
        upper_=x
        for i in range(self.residual_num):
            upper_=self.upper[i](upper_)

        down1_=self.down1(x)
        for i in range(self.residual_num):
            down1_residual_=self.down1_residual[i](down1_)

        if self.sampling_num>1:
            down2_=self.down2(down1_residual_)
            down2_residual_=down2_
        else:
            for i in range(self.residual_num):
                down2_residual_=self.down2_residual[i](down1_residual_)

        for i in range(self.residual_num):
            down3_residual_=self.down3_residual[i](down2_residual_)

        upsampling_=self.upsampling(down3_residual_)

        return upper_+upsampling_