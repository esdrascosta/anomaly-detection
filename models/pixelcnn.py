import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MaskedConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, mask_type, *args, **kwargs):
        self.mask_type = mask_type
        assert mask_type in ['A', 'B'], "Unknown Mask Type"
        super(MaskedConvTranspose2d, self).__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())

        _, depth, height, width = self.weight.size()
        self.mask.fill_(1)
        if mask_type == 'A':
            self.mask[:, :, height // 2, width // 2:] = 0
            self.mask[:, :, height // 2 + 1:, :] = 0
        else:
            self.mask[:, :, height // 2, width // 2 + 1:] = 0
            self.mask[:, :, height // 2 + 1:, :] = 0

    def forward(self, x):

        self.weight.data *= self.mask
        return super(MaskedConvTranspose2d, self).forward(x)


class MaskedCNN(nn.Conv2d):
    """
    Implementation of Masked CNN Class as explained in A Oord et. al.
    Taken from https://github.com/jzbontar/pixelcnn-pytorch
    """

    def __init__(self, mask_type, *args, **kwargs):
        self.mask_type = mask_type
        assert mask_type in ['A', 'B'], "Unknown Mask Type"
        super(MaskedCNN, self).__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())

        _, depth, height, width = self.weight.size()
        self.mask.fill_(1)
        if mask_type == 'A':
            self.mask[:, :, height // 2, width // 2:] = 0
            self.mask[:, :, height // 2 + 1:, :] = 0
        else:
            self.mask[:, :, height // 2, width // 2 + 1:] = 0
            self.mask[:, :, height // 2 + 1:, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedCNN, self).forward(x)


class MaskedConvTransposeBlock(nn.Module):
    def __init__(self, mask_type, input_channels, out_channels, kernel_size, stride, padding, bias):
        super(MaskedConvTransposeBlock, self).__init__()
        self.conv = MaskedConvTranspose2d(mask_type, input_channels, out_channels, kernel_size, stride, padding, bias=bias)
        # self.bn = nn.BatchNorm2d(out_channels)
        self.rl = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        x = self.rl(x)
        return x


class MaskedConvBlock(nn.Module):
    def __init__(self, mask_type, input_channels, out_channels, kernel_size, stride, padding, bias):
        super(MaskedConvBlock, self).__init__()
        self.conv = MaskedCNN(mask_type, input_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.rl = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.rl(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, mask_type, input_channels, out_channels, kernel_size, stride, padding, bias, density=2):
        super(ResidualBlock, self).__init__()
        self.masked_convs_bls = nn.Sequential(
            *[MaskedConvBlock(mask_type, input_channels, out_channels, kernel_size, stride, padding, bias)
              for _ in range(density)]
        )

        self.conv = MaskedCNN(mask_type, input_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.rl = nn.ReLU(True)

    def forward(self, x):
        identity = x
        out = self.masked_convs_bls(x)

        out = self.conv(out)
        out = self.bn(out)

        out += identity
        out = self.rl(out)
        return out


class PixelCNN(nn.Module):
    """
    Network of PixelCNN as described in A Oord et. al.
    """

    def __init__(self, no_layers=8, kernel=7, channels=64, device=None, input_channels=1):
        super(PixelCNN, self).__init__()
        self.no_layers = no_layers
        self.kernel = kernel
        self.channels = channels
        self.layers = {}
        self.device = device

        self.first_conv = MaskedConvBlock('A', input_channels, channels, kernel, 1, kernel // 2, bias=False)

        self.hidden_layers = nn.Sequential(
            *[MaskedConvBlock('B', channels, channels, kernel, 1, kernel // 2, bias=False)\
              for _ in range(self.no_layers)]
        )

        self.out = MaskedCNN('B', channels, input_channels, 1)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.hidden_layers(x)
        return th.sigmoid(self.out(x))


class PixelCNNGenerator(nn.Module):
    def __init__(self, in_channels=32, nc=1, ngf=64):
        super(PixelCNNGenerator, self).__init__()
        self.conv1 = MaskedConvTransposeBlock('A', in_channels, ngf * 4,  4, 1, 0, bias=True)
        self.conv2 = MaskedConvTransposeBlock('B', ngf * 4, ngf * 2,  4, 2, 1, bias=True)
        self.conv3 = MaskedConvTransposeBlock('B', ngf * 2, ngf, 4, 2, 1, bias=True)
        self.conv4 = MaskedConvTransposeBlock('B', ngf, nc, 4, 2, 1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


# only for test purpose
if __name__ == '__main__':
    sample = th.randn(1, 3, 256, 256)
    model = PixelCNN(input_channels=3)
    # model = PixelCNNGenerator(in_channels=100)
    result = model(sample)
    print(f"result={result.size()}")
