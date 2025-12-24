import torch
import torch.nn as nn
import torch.nn.functional as F

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(
                channels, channels, kernel_size, 1,
                padding=get_padding(kernel_size, d), dilation=d
            )) for d in dilations
        ])
        self.convs2 = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(
                channels, channels, kernel_size, 1,
                padding=get_padding(kernel_size, 1), dilation=1
            )) for _ in dilations
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride):
        super().__init__()
        self.conv = nn.utils.weight_norm(
            nn.ConvTranspose1d(
                in_ch, out_ch, kernel, stride,
                padding=(kernel - stride) // 2
            )
        )

    def forward(self, x):
        return self.conv(x)
