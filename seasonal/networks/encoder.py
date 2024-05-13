import torch.nn as nn

EPS = 1e-7

class Encoder(nn.Module):
    def __init__(self, channels_in, channels_out, num_filters=64):
        super(Encoder, self).__init__()

        network = [
            nn.Conv2d(channels_in, num_filters, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.ReLU(inplace=True),

            nn.Conv2d(num_filters, num_filters*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.ReLU(inplace=True),

            nn.Conv2d(num_filters*2, num_filters*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.ReLU(inplace=True),

            nn.Conv2d(num_filters*4, num_filters*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.ReLU(inplace=True),

            nn.Conv2d(num_filters*8, num_filters*8, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True),

            nn.Conv2d(num_filters*8, channels_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        ]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input).reshape(input.size(0), -1) # reshape input to 4D tensor
