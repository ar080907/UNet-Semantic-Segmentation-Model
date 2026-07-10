import torch
import torch.nn as nn
class doubleconv(nn.Module):
    def __init__(self,output_channel,input_channel):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_channel,output_channel,kernel_size=3,padding=1),nn.BatchNorm2d(output_channel),nn.ReLU(inplace=True),nn.Conv2d(output_channel,output_channel,kernel_size=3,padding=1),nn.BatchNorm2d(output_channel),nn.ReLU(inplace=True))
    def forward(self,x):
        return self.conv(x)
class down(nn.Module):
    def __init__(self,output_channel,input_channel):
        super().__init__()
        self.doubleconv = doubleconv(output_channel,input_channel)
        self.downp = nn.MaxPool2d(2)
    def forward(self,x):
        x = self.doubleconv(x)
        skip = x
        x = self.downp(x)
        return x, skip
class up(nn.Module):
    def __init__(self,input_channel,output_channel):
        super().__init__()
        self.tr=nn.ConvTranspose2d(input_channel,output_channel,kernel_size=2,stride=2)
        self.doubleconv = doubleconv(output_channel,output_channel*2)
    def forward(self,x,skip):
        x = self.tr(x)
        x = torch.cat((x,skip),dim=1)
        x = self.doubleconv(x)
        return x
class UNet(nn.Module):
    def __init__(self,input_channel=3,output_channel =1):
        super().__init__()
        self.down1 = down(64,input_channel)
        self.down2 = down(128,64)
        self.down3 = down(256,128)
        self.down4 = down(512,256)
        self.bottleneck = doubleconv(1024,512)
        self.up1 = up(1024,512) 
        self.up2 = up(512,256)
        self.up3 = up(256,128)
        self.up4 = up(128,64)
        self.final = nn.Conv2d(64,output_channel,kernel_size=1)
    def forward(self,x):
        x,skip1 = self.down1(x)
        x,skip2 = self.down2(x)
        x,skip3 = self.down3(x)
        x,skip4 = self.down4(x)
        x = self.bottleneck(x)
        x = self.up1(x,skip4)
        x = self.up2(x,skip3)
        x = self.up3(x,skip2)
        x = self.up4(x,skip1)
        x = self.final(x)
        return x
