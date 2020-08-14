import torch
import torch.nn as nn
import torch.nn.functional as F

from unet.unet_parts import *

class UNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, bilinear=False):
        super(UNet, self).__init__()
        # layers for CT
        self.n_channels = input_size
        self.bilinear = bilinear
        self.sz = 64

        self.inc = DoubleConv3(input_size, hidden_size)
        self.down1 = Down3(hidden_size, hidden_size * 2, self.sz)
        self.down2 = Down3(hidden_size * 2, hidden_size * 4, self.sz // 2)
        self.down3 = Down3(hidden_size * 4, hidden_size * 8, self.sz // 4)
        factor = 2 if bilinear else 1

        self.up1 = Up3(hidden_size * 8, hidden_size * 4 // factor)
        self.up2 = Up3(hidden_size * 4, hidden_size * 2 // factor, bilinear)
        self.up3 = Up3(hidden_size * 2, hidden_size, bilinear)
        self.outc = OutConv3(hidden_size, 1)

        # # layers for X-ray
        self.inc_x = DoubleConv(input_size, hidden_size)
        self.down1_x = Down(hidden_size, hidden_size * 2, self.sz)
        self.down2_x = Down(hidden_size * 2, hidden_size * 4, self.sz // 2)
        self.down3_x = Down(hidden_size * 4, hidden_size * 8, self.sz // 4)
        factor = 2 if bilinear else 1

        self.up1_x = Up(hidden_size * 8, hidden_size * 4 // factor)
        self.up2_x = Up(hidden_size * 4, hidden_size * 2 // factor, bilinear)
        self.up3_x = Up(hidden_size * 2, hidden_size, bilinear)
        self.outc_x = OutConv(hidden_size, 1)

        self.fc1 = nn.Linear(128*128*129, 128*128)
        self.fc2 = nn.Linear(128 * 128, 1024)
        self.fc3 = nn.Linear(1024, 6)


    def forward(self, x, xray):
        # CT
        x = F.interpolate(x, (128, 128, 128))
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)

        # X-ray
        xray = F.interpolate(xray, (xray.shape[2] // 2, xray.shape[3] // 2))
        x1 = self.inc_x(xray)
        x2 = self.down1_x(x1)
        x3 = self.down2_x(x2)
        x4 = self.down3_x(x3)

        x = self.up1_x(x4, x3)
        x = self.up2_x(x, x2)
        x = self.up3_x(x, x1)
        logits_x = self.outc_x(x)

        v = torch.cat([logits.view(-1), logits_x.view(-1)])
        out = self.fc1(v)
        out = self.fc2(out)
        out = self.fc3(out)

        return logits


class Net_split(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net_split, self).__init__()
        # layers for CT
        self.map1 = nn.Conv3d(input_size, hidden_size, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(hidden_size)
        self.map2 = nn.Conv3d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(hidden_size)
        self.map3 = nn.Conv3d(hidden_size, hidden_size*2, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(hidden_size*2)
        self.map4 = nn.Conv3d(hidden_size*2, hidden_size*2, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm3d(hidden_size*2)
        self.map5 = nn.Conv3d(hidden_size * 2, hidden_size * 4, 3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm3d(hidden_size*4)
        self.map6 = nn.Conv3d(hidden_size * 4, hidden_size * 4, 3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm3d(hidden_size*4)
        self.map7 = nn.Conv3d(hidden_size * 4, hidden_size * 8, 3, padding=1, bias=False)
        self.bn7 = nn.BatchNorm3d(hidden_size*8)
        self.map8 = nn.Conv3d(hidden_size * 8, hidden_size * 8, 3, padding=1, bias=False)
        self.bn8 = nn.BatchNorm3d(hidden_size*8)
        self.fc_ct = nn.Linear(hidden_size*8*8*8*8, output_size*2)

        # layers for X-ray
        self.conv1 = nn.Conv2d(input_size, hidden_size, 3, padding=1)
        self.bn1_x = nn.BatchNorm2d(hidden_size)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.bn2_x = nn.BatchNorm2d(hidden_size)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size*2, 3, padding=1)
        self.bn3_x = nn.BatchNorm2d(hidden_size*2)
        self.conv4 = nn.Conv2d(hidden_size*2, hidden_size * 2, 3, padding=1)
        self.bn4_x = nn.BatchNorm2d(hidden_size*2)
        self.conv5 = nn.Conv2d(hidden_size * 2, hidden_size * 4, 3, padding=1)
        self.bn5_x = nn.BatchNorm2d(hidden_size*4)
        self.conv6 = nn.Conv2d(hidden_size * 4, hidden_size*4, 3, padding=1)
        self.bn6_x = nn.BatchNorm2d(hidden_size*4)
        self.conv7 = nn.Conv2d(hidden_size * 4, hidden_size*8, 3, padding=1)
        self.bn7_x = nn.BatchNorm2d(hidden_size*8)
        self.conv8 = nn.Conv2d(hidden_size * 8, hidden_size*8, 3, padding=1)
        self.bn8_x = nn.BatchNorm2d(hidden_size*8)

        self.fc_x = nn.Linear(hidden_size * 8 * 8 * 8, output_size * 2)
        self.fc1 = nn.Linear(output_size * 4, output_size * 3)
        self.fc2 = nn.Linear(output_size * 3, output_size * 2)
        self.fc3 = nn.Linear(output_size * 2, output_size)
        self.fc_out = nn.Linear(output_size, output_size)


    def forward(self, x, xray):
        # CT

        x = F.interpolate(x, (128, 128, 128))
        out = self.map1(x)
        out = F.relu(self.bn1(out))
        out = nn.AdaptiveMaxPool3d((128, 128, 128))(out)

        out = self.map2(out)
        out = F.relu(self.bn2(out))
        out = nn.AdaptiveMaxPool3d((64, 64, 64))(out)

        out = self.map3(out)
        out = F.relu(self.bn3(out))
        out = nn.AdaptiveMaxPool3d((64, 64, 64))(out)

        out = self.map4(out)
        out = F.relu(self.bn4(out))
        out = nn.AdaptiveMaxPool3d((32, 32, 32))(out)

        out = self.map5(out)
        out = F.relu(self.bn5(out))
        out = nn.AdaptiveMaxPool3d((16, 16, 16))(out)

        out = self.map6(out)
        out = F.relu(self.bn6(out))
        out = nn.AdaptiveMaxPool3d((8, 8, 8))(out)

        out = self.map7(out)
        out = F.relu(self.bn7(out))
        out = nn.AdaptiveMaxPool3d((8, 8, 8))(out)

        out = self.map8(out)
        out = F.relu(self.bn8(out))
        out = nn.AdaptiveMaxPool3d((8, 8, 8))(out)

        out = out.view((1, -1))
        out_ct = F.relu(self.fc_ct(out))

        # X-ray
        xray = F.interpolate(xray, (xray.shape[2] // 2, xray.shape[3] // 2))
        out = self.conv1(xray)
        out = F.relu(self.bn1_x(out))
        out = nn.AdaptiveMaxPool2d((120, 155))(out)

        out = self.conv2(out)
        out = F.relu(self.bn2_x(out))
        out = nn.AdaptiveMaxPool2d((64, 64))(out)

        out = self.conv3(out)
        out = F.relu(self.bn3_x(out))
        out = nn.AdaptiveMaxPool2d((32, 32))(out)

        out = self.conv4(out)
        out = F.relu(self.bn4_x(out))
        out = nn.AdaptiveMaxPool2d((32, 32))(out)

        out = self.conv5(out)
        out = F.relu(self.bn5_x(out))
        out = nn.AdaptiveMaxPool2d((16, 16))(out)

        out = self.conv6(out)
        out = F.relu(self.bn6_x(out))
        out = nn.AdaptiveMaxPool2d((8, 8))(out)

        out = self.conv7(out)
        out = F.relu(self.bn7_x(out))
        out = nn.AdaptiveMaxPool2d((8, 8))(out)

        out = self.conv8(out)
        out = F.relu(self.bn8_x(out))
        out = nn.AdaptiveMaxPool2d((8, 8))(out)

        out = out.view((1, -1))
        out_xray = F.relu(self.fc_x(out))

        out = torch.cat((out_ct, out_xray), dim=-1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc_out(out)



        return out


class layer6Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(layer6Net, self).__init__()
        # layers for CT
        self.map1 = nn.Conv3d(input_size, hidden_size, 5, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(hidden_size)
        self.map2 = nn.Conv3d(hidden_size, hidden_size, 5, padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(hidden_size)
        self.map3 = nn.Conv3d(hidden_size, hidden_size, 5, padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(hidden_size)
        self.fc_ct = nn.Linear(hidden_size*32*32*32, output_size*2)

        # layers for X-ray
        self.conv1 = nn.Conv2d(input_size, hidden_size, 3, padding=1)
        self.bn1_x = nn.BatchNorm2d(hidden_size)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.bn2_x = nn.BatchNorm2d(hidden_size)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.bn3_x = nn.BatchNorm2d(hidden_size*2)


        self.fc_x = nn.Linear(hidden_size * 32 * 32, output_size * 2)
        self.fc1 = nn.Linear(output_size * 4, output_size * 3)
        self.fc2 = nn.Linear(output_size * 3, output_size * 2)
        self.fc3 = nn.Linear(output_size * 2, output_size)
        self.fc_out = nn.Linear(output_size, output_size)


    def forward(self, x, xray):
        # CT
        x = F.interpolate(x, (128, 128, 128))
        out = self.map1(x)
        # out = F.relu(self.bn1(out))
        out = F.relu(out)
        out = nn.AdaptiveMaxPool3d((128, 128, 128))(out)

        out = self.map2(out)
        # out = F.relu(self.bn2(out))
        out = F.relu(out)
        out = nn.AdaptiveMaxPool3d((64, 64, 64))(out)

        out = self.map3(out)
        # out = F.relu(self.bn3(out))
        out = F.relu(out)
        out = nn.AdaptiveMaxPool3d((32, 32, 32))(out)

        out = out.view((1, -1))
        out_ct = F.relu(self.fc_ct(out))

        # X-ray
        xray = F.interpolate(xray, (xray.shape[2]//2, xray.shape[3]//2))
        out = self.conv1(xray)
        # out = F.relu(self.bn1_x(out))
        out = F.relu(out)
        out = nn.AdaptiveMaxPool2d((128, 128))(out)

        out = self.conv2(out)
        # out = F.relu(self.bn2_x(out))
        out = F.relu(out)
        out = nn.AdaptiveMaxPool2d((64, 64))(out)

        out = self.conv3(out)
        # out = F.relu(self.bn3_x(out))
        out = F.relu(out)
        out = nn.AdaptiveMaxPool2d((32, 32))(out)


        out = out.view((1, -1))
        out_xray = F.relu(self.fc_x(out))

        out = torch.cat((out_ct, out_xray), dim=-1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc_out(out)



        return out

if __name__ == "__main__":
    # The number of block is 5.
    net = Net(3, 16, 6)