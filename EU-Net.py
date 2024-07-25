import torch
import torch.nn as nn
import torch.functional as F
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.5):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1 )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_prob)
        self.sigmoid = nn.Sigmoid()
        self.match_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
                              if in_channels != out_channels else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.match_channels is not None:
            identity = self.match_channels(identity)
        B = identity*self.sigmoid(identity)
        out += B
        out = self.relu(out)
        out = self.dropout(out)

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class UNet(nn.Module):
    def __init__(self, input_nc, output_nc,dropout_prob=0.5):
        super(UNet, self).__init__()

        # Encoding path
        self.in_block = ResidualBlock(input_nc, 64)
        self.dilated_conv_in_block = nn.Conv2d(64 , 64, kernel_size=3, padding=4, dilation=4)
        self.down1 = ResidualBlock(64, 128)
        self.dilated_conv_down1 = nn.Conv2d(128, 128, kernel_size=3, padding=3, dilation=3)
        self.down2 = ResidualBlock(128, 256)
        self.dilated_conv_down2 = nn.Conv2d(256, 256, kernel_size=3, padding=2, dilation=2)
        self.down3 = ResidualBlock(256, 512)
        self.down4 = ResidualBlock(512, 1024)
        # Pooling
        self.pooling = nn.MaxPool2d(2)

        # Decoding path
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_block1 = ResidualBlock(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_block2 = ResidualBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_block3 = ResidualBlock(256, 128)
        self.dilated_conv_up_block2 = nn.Conv2d(256, 256, kernel_size=3, padding=2, dilation=2)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dilated_conv_up_block3 = nn.Conv2d(128, 128, kernel_size=3, padding=3, dilation=3)
        self.up_block4 = ResidualBlock(128, 64)
        self.CA_down1 = ChannelAttention(128)
        self.SA_up4 = SpatialAttention(kernel_size=3)
        self.dilated_conv_up_block4  = nn.Conv2d(64, 64, kernel_size=3, padding=4, dilation=4)
        self.dropout = nn.Dropout(dropout_prob)
        # Output layer
        self.out_conv = nn.Conv2d(192, output_nc, kernel_size=1)

    def forward(self, x):
        # Encoding path
        x1 = self.in_block(x)
        x1 = self.dilated_conv_in_block(x1)
        x2 = self.down1(self.pooling(x1))
        x2 = self.dilated_conv_down1(x2)
        x2 = self.CA_down1(x2) * x2
        x3 = self.down2(self.pooling(x2))
        x3 = self.dilated_conv_down2(x3)
        x4 = self.down3(self.pooling(x3))
        x5 = self.down4(self.pooling(x4))


        # Decoding path
        x = self.up1(x5)
        x = self.dropout(x)
        x = self.up_block1(torch.cat([x, x4], dim=1))
        x = self.up2(x)
        x = self.dropout(x)
        x = self.up_block2(torch.cat([x, x3], dim=1))
        x = self.dilated_conv_up_block2(x)
        x = self.up3(x)
        x = self.dropout(x)
        x = self.up_block3(torch.cat([x, x2], dim=1))
        x = self.dilated_conv_up_block3(x)
        x = self.up4(x)
        x = self.dropout(x)
        x_SA = self.SA_up4(x) * x
        x_up_block4 = self.up_block4(torch.cat([x, x1], dim=1))
        x_dilated = self.dilated_conv_up_block4(x_up_block4)
        x_combined = torch.cat([x_SA, x_up_block4, x_dilated], dim=1)

        return self.out_conv(x_combined)


