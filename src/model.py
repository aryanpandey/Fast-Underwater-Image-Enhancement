import torch
import torch.nn as nn
import math


class conv_block(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.Dropout2d(0.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.Dropout2d(0.2),
            nn.ReLU(inplace=True)
        )

        self.final_block = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels-3, kernel_size=3, padding = 1),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        out = self.block(x)
        out = (out + x)/math.sqrt(2)
        out = self.final_block(out)
        return out

class Shallow_UWNet(nn.Module):
    def __init__(self, conv_in, conv_mid, network_depth = 2):
        super().__init__()

        self.initial_network = nn.Sequential(
            nn.Conv2d(3, conv_in, kernel_size=3, padding = 1),
            nn.ReLU(inplace=True),
            conv_block(conv_in, conv_mid)
        )
        
        self.conv_blocks = nn.ModuleList()
        for i in range(network_depth):
            self.conv_blocks.append(conv_block(conv_mid, conv_mid))

        self.output_conv = nn.Conv2d(conv_mid, 3, kernel_size=3, padding = 1)

    def forward(self, x):
        out = self.initial_network(x)
        out = torch.cat([out, x], dim = 1)

        for block in self.conv_blocks:
            out = block(out)
            out = torch.cat([out, x], dim = 1)
        
        out = self.output_conv(out)

        return out

