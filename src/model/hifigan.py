import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.modules import ResBlock, UpsampleBlock, get_padding

class Generator(nn.Module):
    def __init__(self, n_mels, upsample_rates, upsample_kernels, resblock_kernel_sizes, resblock_dilations, channels=512):
        super().__init__()
        self.pre = nn.utils.weight_norm(nn.Conv1d(n_mels, channels, 7, 1, padding=3))

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        cur_ch = channels
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernels)):
            self.ups.append(UpsampleBlock(cur_ch, cur_ch // 2, k, u))
            cur_ch = cur_ch // 2

            for (rk, rd) in zip(resblock_kernel_sizes, resblock_dilations):
                self.resblocks.append(ResBlock(cur_ch, kernel_size=rk, dilations=rd))

        self.post = nn.utils.weight_norm(nn.Conv1d(cur_ch, 1, 7, 1, padding=3))

        self.num_res_per_up = len(resblock_kernel_sizes)

    def forward(self, mel):
        x = self.pre(mel)
        rb_i = 0
        for up in self.ups:
            x = F.leaky_relu(x, 0.1)
            x = up(x)

            xs = 0
            for _ in range(self.num_res_per_up):
                xs = xs + self.resblocks[rb_i](x)
                rb_i += 1
            x = xs / self.num_res_per_up

        x = F.leaky_relu(x, 0.1)
        x = self.post(x)
        x = torch.tanh(x)
        return x


class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, channels=32):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv2d(1, channels, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size), 0))),
            nn.utils.weight_norm(nn.Conv2d(channels, channels*4, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size), 0))),
            nn.utils.weight_norm(nn.Conv2d(channels*4, channels*16, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size), 0))),
            nn.utils.weight_norm(nn.Conv2d(channels*16, channels*16, (kernel_size, 1), 1, padding=(get_padding(kernel_size), 0))),
        ])
        self.post = nn.utils.weight_norm(nn.Conv2d(channels*16, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        b, c, t = x.shape
        if t % self.period != 0:
            pad = self.period - (t % self.period)
            x = F.pad(x, (0, pad), "reflect")
            t = t + pad
        x = x.view(b, c, t // self.period, self.period)

        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2,3,5,7,11)):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(p) for p in periods])

    def forward(self, x):
        outs, fmaps = [], []
        for d in self.discriminators:
            o, f = d(x)
            outs.append(o); fmaps.append(f)
        return outs, fmaps

class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.avgpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, x):
        outs, fmaps = [], []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                x = self.avgpools[i-1](x)
            o, f = d(x)
            outs.append(o); fmaps.append(f)
        return outs, fmaps
