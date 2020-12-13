import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction

from unet.unet_parts import *

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

def double_conv3(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):

    def __init__(self, input, hidden, n_class):
        super().__init__()

        self.dconv_down1 = double_conv3(input, hidden)
        self.dconv_down2 = double_conv3(hidden, hidden*2)
        self.dconv_down3 = double_conv3(hidden*2, hidden*4)
        self.dconv_down4 = double_conv3(hidden*4, hidden*8)

        self.maxpool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.dconv_up3 = double_conv3(hidden*4 + hidden*8, hidden*4)
        self.dconv_up2 = double_conv3(hidden*2 + hidden*4, hidden*2)
        self.dconv_up1 = double_conv3(hidden*2 + hidden, hidden)

        self.conv_last = nn.Conv3d(hidden, 1, 1)
        self.fc_ct = nn.Linear(32*32*32, 128)

        self.dconv_down1_x = double_conv(input, hidden)
        self.dconv_down2_x = double_conv(hidden, hidden*2)
        self.dconv_down3_x = double_conv(hidden*2, hidden*4)
        self.dconv_down4_x = double_conv(hidden*4, hidden*8)

        self.maxpool_x = nn.MaxPool2d(2)
        self.upsample_x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3_x = double_conv(hidden*4 + hidden*8, hidden*4)
        self.dconv_up2_x = double_conv(hidden*2 + hidden*4, hidden*2)
        self.dconv_up1_x = double_conv(hidden*2 + hidden, hidden)

        self.conv_last_x = nn.Conv2d(hidden, 1, 1)
        self.fc_x = nn.Linear(32 * 32, 128)


        self.fc = nn.Linear(256, 6)

    def forward(self, x, xray):
        x = F.interpolate(x, (64, 64, 64))
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = F.relu(self.conv_last(x))
        out = self.maxpool(out)
        out = self.fc_ct(out.view((1, -1)))



        # X-ray
        xray = F.interpolate(xray, (64, 64))
        conv1 = self.dconv_down1_x(xray)
        x = self.maxpool_x(conv1)

        conv2 = self.dconv_down2_x(x)
        x = self.maxpool_x(conv2)

        conv3 = self.dconv_down3_x(x)
        x = self.maxpool_x(conv3)

        x = self.dconv_down4_x(x)

        x = self.upsample_x(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3_x(x)
        x = self.upsample_x(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2_x(x)
        x = self.upsample_x(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1_x(x)

        out_x = F.relu(self.conv_last_x(x))
        out_x = self.maxpool_x(out_x)
        out_x = self.fc_x(out_x.view((1, -1)))

        out = torch.cat((out, out_x), dim=-1)
        out = self.fc(out)



        return out


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
        self.fc_ct = nn.Linear(hidden_size*16*16*16, output_size*2)

        # layers for X-ray
        self.conv1 = nn.Conv2d(input_size, hidden_size, 3, padding=1)
        self.bn1_x = nn.BatchNorm2d(hidden_size)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.bn2_x = nn.BatchNorm2d(hidden_size)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.bn3_x = nn.BatchNorm2d(hidden_size*2)


        self.fc_x = nn.Linear(hidden_size * 16 * 16, output_size * 2)
        self.fc1 = nn.Linear(output_size * 4, output_size * 3)
        self.fc2 = nn.Linear(output_size * 3, output_size * 2)
        self.fc3 = nn.Linear(output_size * 2, output_size)
        self.fc_out = nn.Linear(output_size, output_size)


    def forward(self, x, xray):
        # CT
        x = F.interpolate(x, (64, 64, 64))
        out = self.map1(x)
        # out = F.relu(self.bn1(out))
        out = F.relu(out)
        out = nn.AdaptiveMaxPool3d((64, 64, 64))(out)

        out = self.map2(out)
        # out = F.relu(self.bn2(out))
        out = F.relu(out)
        out = nn.AdaptiveMaxPool3d((32, 32, 32))(out)

        out = self.map3(out)
        # out = F.relu(self.bn3(out))
        out = F.relu(out)
        out = nn.AdaptiveMaxPool3d((16, 16, 16))(out)

        out = out.view((1, -1))
        out_ct = F.relu(self.fc_ct(out))

        # X-ray
        xray = F.interpolate(xray, (xray.shape[2]//2, xray.shape[3]//2))
        out = self.conv1(xray)
        # out = F.relu(self.bn1_x(out))
        out = F.relu(out)
        out = nn.AdaptiveMaxPool2d((64, 64))(out)

        out = self.conv2(out)
        # out = F.relu(self.bn2_x(out))
        out = F.relu(out)
        out = nn.AdaptiveMaxPool2d((32, 32))(out)

        out = self.conv3(out)
        # out = F.relu(self.bn3_x(out))
        out = F.relu(out)
        out = nn.AdaptiveMaxPool2d((16, 16))(out)


        out = out.view((1, -1))
        out_xray = F.relu(self.fc_x(out))

        out = torch.cat((out_ct, out_xray), dim=-1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc_out(out)



        return out


class HomographyNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(HomographyNet, self).__init__()
        # layers for CT
        self.map1 = nn.Conv3d(input_size, hidden_size, 5, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(hidden_size)
        self.map2 = nn.Conv3d(hidden_size, hidden_size, 5, padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(hidden_size)
        self.map3 = nn.Conv3d(hidden_size, hidden_size, 5, padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(hidden_size)
        self.map4 = nn.Conv3d(hidden_size, hidden_size, 5, padding=0, bias=False)
        self.bn4 = nn.BatchNorm3d(hidden_size)
        self.map5 = nn.Conv3d(hidden_size, hidden_size, 5, padding=0, bias=False)
        self.bn5 = nn.BatchNorm3d(hidden_size)
        self.map6 = nn.Conv3d(hidden_size, hidden_size, 5, padding=0, bias=False)
        self.bn6 = nn.BatchNorm3d(hidden_size)
        self.fc_ct = nn.Linear(hidden_size * 16 * 16 * 16, 512)

        # layers for X-ray
        self.conv1 = nn.Conv2d(input_size, hidden_size, 3, padding=1)
        self.bn1_x = nn.BatchNorm2d(hidden_size)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.bn2_x = nn.BatchNorm2d(hidden_size)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.bn3_x = nn.BatchNorm2d(hidden_size)
        self.conv4 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.bn4_x = nn.BatchNorm2d(hidden_size)
        self.conv5 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.bn5_x = nn.BatchNorm2d(hidden_size)
        self.conv6 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.bn6_x = nn.BatchNorm2d(hidden_size)

        self.fc_x = nn.Linear(hidden_size * 16 * 16, 512)
        self.fc_out = nn.Linear(1024, output_size)

    def forward(self, x, xray):
        # CT
        x = F.interpolate(x, (64, 64, 64))
        out = self.map1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.map2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = nn.AdaptiveMaxPool3d((64, 64, 64))(out)

        out = self.map3(out)
        out = self.bn3(out)
        out = F.relu(out)

        out = self.map4(out)
        out = self.bn4(out)
        out = F.relu(out)
        out = nn.AdaptiveMaxPool3d((32, 32, 32))(out)

        out = self.map5(out)
        out = self.bn5(out)
        out = F.relu(out)

        out = self.map6(out)
        out = self.bn6(out)
        out = F.relu(out)
        out = nn.AdaptiveMaxPool3d((16, 16, 16))(out)

        out = out.view((1, -1))
        out_ct = F.relu(self.fc_ct(out))

        # X-ray
        xray = F.interpolate(xray, (xray.shape[2] // 2, xray.shape[3] // 2))
        out = self.conv1(xray)
        out = self.bn1_x(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2_x(out)
        out = F.relu(out)
        out = nn.AdaptiveMaxPool2d((64, 64))(out)

        out = self.conv3(out)
        out = self.bn3_x(out)
        out = F.relu(out)
        out = self.conv4(out)
        out = self.bn4_x(out)
        out = F.relu(out)
        out = nn.AdaptiveMaxPool2d((32, 32))(out)

        out = self.conv5(out)
        out = self.bn5_x(out)
        out = F.relu(out)
        out = self.conv6(out)
        out = self.bn6_x(out)
        out = F.relu(out)
        out = nn.AdaptiveMaxPool2d((16, 16))(out)

        out = out.view((1, -1))
        out_xray = F.relu(self.fc_x(out))

        out = torch.cat((out_ct, out_xray), dim=-1)
        out = self.fc_out(out)



        return out


class HomographyNet_bn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(HomographyNet_bn, self).__init__()
        # layers for CT
        self.map1 = nn.Conv3d(input_size, hidden_size, 5, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(hidden_size)
        self.map2 = nn.Conv3d(hidden_size, hidden_size, 5, padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(hidden_size)
        self.map3 = nn.Conv3d(hidden_size, hidden_size, 5, padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(hidden_size)
        self.map4 = nn.Conv3d(hidden_size, hidden_size, 5, padding=0, bias=False)
        self.bn4 = nn.BatchNorm3d(hidden_size)
        self.map5 = nn.Conv3d(hidden_size, hidden_size, 5, padding=0, bias=False)
        self.bn5 = nn.BatchNorm3d(hidden_size)
        self.map6 = nn.Conv3d(hidden_size, hidden_size, 5, padding=0, bias=False)
        self.bn6 = nn.BatchNorm3d(hidden_size)
        self.fc_ct = nn.Linear(hidden_size*16*16*16, output_size*2)

        # layers for X-ray
        self.conv1 = nn.Conv2d(input_size, hidden_size, 3, padding=1)
        self.bn1_x = nn.BatchNorm2d(hidden_size)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.bn2_x = nn.BatchNorm2d(hidden_size)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.bn3_x = nn.BatchNorm2d(hidden_size)
        self.conv4 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.bn4_x = nn.BatchNorm2d(hidden_size)
        self.conv5 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.bn5_x = nn.BatchNorm2d(hidden_size)
        self.conv6 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.bn6_x = nn.BatchNorm2d(hidden_size)


        self.fc_x = nn.Linear(hidden_size * 16 * 16, output_size * 2)
        self.fc1 = nn.Linear(output_size * 4, output_size * 2)
        self.fc2 = nn.Linear(output_size * 2, output_size)
        self.fc_out = nn.Linear(output_size, output_size)


    def forward(self, x, xray):
        # CT
        x = F.interpolate(x, (64, 64, 64))
        out = self.map1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.map2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = nn.AdaptiveMaxPool3d((64, 64, 64))(out)

        out = self.map3(out)
        out = self.bn3(out)
        out = F.relu(out)

        out = self.map4(out)
        out = self.bn4(out)
        out = F.relu(out)
        out = nn.AdaptiveMaxPool3d((32, 32, 32))(out)

        out = self.map5(out)
        out = self.bn5(out)
        out = F.relu(out)

        out = self.map6(out)
        out = self.bn6(out)
        out = F.relu(out)
        out = nn.AdaptiveMaxPool3d((16, 16, 16))(out)

        out = out.view((1, -1))
        out_ct = F.relu(self.fc_ct(out))

        # X-ray
        xray = F.interpolate(xray, (xray.shape[2]//2, xray.shape[3]//2))
        out = self.conv1(xray)
        out = self.bn1_x(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2_x(out)
        out = F.relu(out)
        out = nn.AdaptiveMaxPool2d((64, 64))(out)

        out = self.conv3(out)
        out = self.bn3_x(out)
        out = F.relu(out)
        out = self.conv4(out)
        out = self.bn4_x(out)
        out = F.relu(out)
        out = nn.AdaptiveMaxPool2d((32, 32))(out)

        out = self.conv5(out)
        out = self.bn5_x(out)
        out = F.relu(out)
        out = self.conv6(out)
        out = self.bn6_x(out)
        out = F.relu(out)
        out = nn.AdaptiveMaxPool2d((16, 16))(out)


        out = out.view((1, -1))
        out_xray = F.relu(self.fc_x(out))

        out = torch.cat((out_ct, out_xray), dim=-1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc_out(out)



        return out


class layer8Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(layer8Net, self).__init__()
        # layers for CT
        self.map1 = nn.Conv3d(input_size, hidden_size*2, 5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm3d(hidden_size*2)
        self.map2 = nn.Conv3d(hidden_size*2, hidden_size*2, 5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm3d(hidden_size*2)
        self.map3 = nn.Conv3d(hidden_size*2, hidden_size*4, 5, padding=2, bias=False)
        self.bn3 = nn.BatchNorm3d(hidden_size*4)
        self.map4 = nn.Conv3d(hidden_size*4, hidden_size*4, 5, padding=2, bias=False)
        self.bn4 = nn.BatchNorm3d(hidden_size*4)
        self.fc_ct = nn.Linear(hidden_size*4*32*32*32, output_size*2)

        # layers for X-ray
        self.conv1 = nn.Conv2d(input_size, hidden_size*2, 3, padding=1)
        self.bn1_x = nn.BatchNorm2d(hidden_size*2)
        self.conv2 = nn.Conv2d(hidden_size*2, hidden_size*2, 3, padding=1)
        self.bn2_x = nn.BatchNorm2d(hidden_size*2)
        self.conv3 = nn.Conv2d(hidden_size*2, hidden_size*4, 3, padding=1)
        self.bn3_x = nn.BatchNorm2d(hidden_size*4)
        self.conv4 = nn.Conv2d(hidden_size*4, hidden_size*4, 3, padding=1)
        self.bn4_x = nn.BatchNorm2d(hidden_size*4)
        self.fc_x = nn.Linear(hidden_size * 4 * 32 * 32, output_size * 2)


        self.fc1 = nn.Linear(output_size * 4, output_size * 3)
        self.fc2 = nn.Linear(output_size * 3, output_size * 2)
        self.fc3 = nn.Linear(output_size * 2, output_size)
        self.fc_out = nn.Linear(output_size, output_size)


    def forward(self, x, xray):
        # CT
        x = F.interpolate(x, (128, 128, 128))
        out = self.map1(x)
        out = F.relu(self.bn1(out))
        # out = F.relu(out)
        out = nn.AdaptiveMaxPool3d((128, 128, 128))(out)

        out = self.map2(out)
        out = F.relu(self.bn2(out))
        # out = F.relu(out)
        out = nn.AdaptiveMaxPool3d((64, 64, 64))(out)

        out = self.map3(out)
        out = F.relu(self.bn3(out))
        # out = F.relu(out)
        out = nn.AdaptiveMaxPool3d((32, 32, 32))(out)

        out = self.map4(out)
        out = F.relu(self.bn4(out))
        # out = F.relu(out)
        out = nn.AdaptiveMaxPool3d((32, 32, 32))(out)

        out = out.view((1, -1))
        out_ct = F.relu(self.fc_ct(out))

        # X-ray
        xray = F.interpolate(xray, (xray.shape[2]//2, xray.shape[3]//2))
        out = self.conv1(xray)
        out = F.relu(self.bn1_x(out))
        # out = F.relu(out)
        out = nn.AdaptiveMaxPool2d((128, 128))(out)

        out = self.conv2(out)
        out = F.relu(self.bn2_x(out))
        # out = F.relu(out)
        out = nn.AdaptiveMaxPool2d((64, 64))(out)

        out = self.conv3(out)
        out = F.relu(self.bn3_x(out))
        # out = F.relu(out)
        out = nn.AdaptiveMaxPool2d((32, 32))(out)

        out = self.conv4(out)
        out = F.relu(self.bn4_x(out))
        # out = F.relu(out)
        out = nn.AdaptiveMaxPool2d((32, 32))(out)


        out = out.view((1, -1))
        out_xray = F.relu(self.fc_x(out))

        out = torch.cat((out_ct, out_xray), dim=-1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc_out(out)



        return out


## PointNet



class PointReg(nn.Module):
    def __init__(self, out_dim, normal_channel=False):
        super(PointReg, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320, [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 256, 512], True)

        self.sa1_x = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], -1, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2_x = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 319, [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3_x = PointNetSetAbstraction(None, None, None, 640 + 2, [256, 256, 512], True)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, out_dim)

    def forward(self, xyz, xy):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
            xy = xy[:, :2, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 512)

        l1_xy, l1_points = self.sa1_x(xy, norm)
        l2_xy, l2_points = self.sa2_x(l1_xy, l1_points)
        l3_xy, l3_points = self.sa3_x(l2_xy, l2_points)
        x_2 = l3_points.view(B, 512)

        x = torch.cat((x, x_2), dim=1)
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x



if __name__ == "__main__":
    # The number of block is 5.
    net = Net(3, 16, 6)