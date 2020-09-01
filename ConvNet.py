import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.fc_ct = nn.Linear(64*64*64, 128)

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
        self.fc_x = nn.Linear(64 * 64, 128)


        self.fc = nn.Linear(256, 6)

    def forward(self, x, xray):
        x = F.interpolate(x, (128, 128, 128))
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
        xray = F.interpolate(xray, (128, 128))
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


if __name__ == "__main__":
    # The number of block is 5.
    net = Net(3, 16, 6)