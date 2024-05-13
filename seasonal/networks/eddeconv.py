import torch.nn as nn

# Encoder-Decoder Deconvolutional Network
class EDDeconv(nn.Module):
    def __init__(self, channels_in, channels_out, zdim=128, num_filters=64):
        super(EDDeconv, self).__init__()
        # Downsampling
        network = [
            nn.Conv2d(channels_in, num_filters, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(16, num_filters),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters, num_filters*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16*2, num_filters*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters*2, num_filters*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.GroupNorm(16*4, num_filters*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters*4, num_filters*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters*8, zdim, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True)]

        # Upsampling
        network += [
            nn.ConvTranspose2d(zdim, num_filters*8, kernel_size=4, stride=1, padding=0, bias=False),  # 1x1 -> 4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters*8, num_filters*8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_filters*8, num_filters*4, kernel_size=4, stride=2, padding=1, bias=False),  # 4x4 -> 8x8
            nn.GroupNorm(16*4, num_filters*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters*4, num_filters*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16*4, num_filters*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_filters*4, num_filters*2, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 16x16
            nn.GroupNorm(16*2, num_filters*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters*2, num_filters*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16*2, num_filters*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_filters*2, num_filters, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 32x32
            nn.GroupNorm(16, num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, num_filters),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32x32 -> 64x64
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(16, num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, channels_out, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Tanh()
        ]

        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input)