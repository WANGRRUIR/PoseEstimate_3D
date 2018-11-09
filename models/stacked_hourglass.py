import torch.nn as nn
from models.residual_module import Residual
from models.hourglass import Hourglass

class StackedHourgalss(nn.Module):
    def __init__(self,stack_num,residual_num,channel_num,output_num):
        super(StackedHourgalss,self).__init__()
        self.stack_num=stack_num
        self.residual_num=residual_num
        self.channel_num=channel_num
        self.output_num=output_num

        self.conv1=nn.Conv2d(3,64,bias=True,kernel_size=7,stride=2,padding=3)
        self.bn=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.rm1=Residual(64,128)
        self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.rm2=Residual(128,128)
        self.rm3=Residual(128,128)
        self.rm4=Residual(128,self.channel_num)

        _hourglass,_residual,_inner_fc,_inner_out,_fc,_out=[],[],[],[],[],[]
        for i in range(stack_num):
            _hourglass.append(Hourglass(4,self.residual_num,self.channel_num))
            for j in range(self.residual_num):
                _residual.append(Residual(self.channel_num,self.channel_num))
            fc=nn.Sequential(
                nn.Conv2d(self.channel_num,self.channel_num,bias=True,kernel_size=1,stride=1),nn.BatchNorm2d(self.channel_num),self.relu
            )
            _fc.append(fc)
            _out.append(nn.Conv2d(self.channel_num,self.output_num,bias=True,kernel_size=1,stride=1))
            if i<self.stack_num-1:
                _inner_fc.append(nn.Conv2d(self.channel_num,self.channel_num,bias=True,kernel_size=1,stride=1))
                _inner_out.append(nn.Conv2d(self.output_num,self.channel_num,bias=True,kernel_size=1,stride=1))

        self.hourglass=nn.ModuleList(_hourglass)
        self.residual=nn.ModuleList(_residual)
        self.inner_fc=nn.ModuleList(_inner_fc)
        self.inner_out=nn.ModuleList(_inner_out)
        self.fc=nn.ModuleList(_fc)
        self.out=nn.ModuleList(_out)

    def forward(self, x):
        x=self.conv1(x)
        x=self.bn(x)
        x=self.relu(x)
        x=self.rm1(x)
        x=self.maxpool(x)
        x=self.rm2(x)
        x=self.rm3(x)
        x=self.rm4(x)

        out_heatmap=[]

        for i in range(self.stack_num):
            hg=self.hourglass[i](x)
            x_remap=hg
            for j in range(self.residual_num):
                x_remap=self.residual[i*self.residual_num+j](x_remap)
            x_remap=self.fc[i](x_remap)
            out_=self.out[i](x_remap)
            out_heatmap.append(out_)
            if i<self.stack_num-1:
                inner_fc_=self.inner_fc[i](x_remap)
                inner_out_=self.inner_out[i](out_)
                x=x+inner_fc_+inner_out_

        return out_heatmap