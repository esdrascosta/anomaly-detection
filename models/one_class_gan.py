# todo USE this code https://github.com/jalola/improved-wgan-pytorch/blob/master/models/wgan.py

import torch
from torch import nn

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64


class Encoder(nn.Module):
    def __init__(self, ngpu=1, channels=1, nz=100):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, nz, 1, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu=1, nc=1):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
    else:
        return nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                             conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0))


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(UNetDownBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv1 = conv(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu1 = nn.LeakyReLU()


    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))

        return x


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, merge_mode='concat', up_mode='transpose'):
        super(UNetUpBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv(2 * self.out_channels, self.out_channels)
        else:
            self.conv1 = conv(self.out_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu1 = nn.LeakyReLU()

    def forward(self, from_up, from_down):
        from_up = self.upconv(from_up)

        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = self.relu1(self.bn1(self.conv1(x)))

        return x


class UNetGenerator(nn.Module):
    def __init__(self, n_channels=1, merge_mode='concat', up_mode='transpose', nz=100):
        super(UNetGenerator, self).__init__()
        self.n_chnnels = n_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.down1 = UNetDownBlock(self.n_chnnels, 64, 3, 1, 1)
        self.down2 = UNetDownBlock(64, 128, 4, 2, 1)
        self.down3 = UNetDownBlock(128, 256, 4, 2, 1)
        self.down4 = UNetDownBlock(256, 512, 4, 2, 1)
        self.down5 = UNetDownBlock(512, nz, 4, 2, 1)

        self.up1 = UNetUpBlock(nz, 512, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.up2 = UNetUpBlock(512, 256, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.up3 = UNetUpBlock(256, 128, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.up4 = UNetUpBlock(128, 64, merge_mode=self.merge_mode, up_mode=self.up_mode)

        self.conv_final = nn.Sequential(conv(64, self.n_chnnels, 3, 1, 1), nn.Tanh())

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        z = self.down5(x4)

        x = self.up1(z, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.conv_final(x)

        return x, z

if __name__ == "__main__":
    unet = Encoder()
    test= torch.randn((1, 1, 64, 64))
    z = unet(test)
    # print(result.shape)
    print(z.shape)