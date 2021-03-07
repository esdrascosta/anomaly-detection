import torch
from torch import nn
from models.pixelcnn import MaskedCNN, ResidualBlock, MaskedConvBlock


class ResidualPixelCNN(nn.Module):
    """
    Network of PixelCNN as described in A Oord et. al.
    """

    def __init__(self, no_layers=8, kernel=7, channels=64, device=None, input_channels=1):
        super(ResidualPixelCNN, self).__init__()
        self.no_layers = no_layers
        self.kernel = kernel
        self.channels = channels
        self.layers = {}
        self.device = device

        self.first_conv = MaskedConvBlock('A', input_channels, channels, kernel, 1, kernel // 2, bias=False)

        self.hidden_layers = nn.Sequential(
            *[ResidualBlock('B', channels, channels, kernel, 1, kernel // 2, bias=False, density=1) for _ in range(self.no_layers)]
        )

        self.out = MaskedCNN('B', channels, 1, 1)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.hidden_layers(x)
        return self.out(x)


# only for test purpose
if __name__ == '__main__':
    device = 'cpu'
    sample = torch.rand(1, 1, 64, 64).to(device)
    model = ResidualPixelCNN().to(device)
    result = model(sample)
    print(f"result={result.size()}")