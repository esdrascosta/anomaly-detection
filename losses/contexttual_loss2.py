from torch.nn.modules.loss import _Loss as Loss
from torchvision.models import vgg16
import torch
import contextual_loss.functional as F


class ContextualLoss2(Loss):
    """
    Creates a criterion that measures the contextual loss.

    Parameters
    ---
    band_width : int, optional
        a band_width parameter described as :math:`h` in the paper.
    use_vgg : bool, optional
        if you want to use VGG feature, set this `True`.
    vgg_layer : str, optional
        intermidiate layer name for VGG feature.
        Now we support layer names:
            `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
    """

    def __init__(self,
                 band_width: float = 0.5,
                 loss_type: str = 'l2',
                 use_vgg: bool = False,
                 vgg_layer: str = 'relu3_4'):

        super(ContextualLoss2, self).__init__()

        assert band_width > 0, 'band_width parameter must be positive.'

        self.band_width = band_width
        self.loss_type= loss_type
        if use_vgg:
            self.vgg_model = vgg16(pretrained=True).eval()

            for param in self.vgg_model.parameters():
                param.requires_grad = False

            self.vgg_layer = vgg_layer
            self.register_buffer(
                name='vgg_mean',
                tensor=torch.tensor(
                    [[[0.485]], [[0.456]], [[0.406]]], requires_grad=False).to('cuda')
            )
            self.register_buffer(
                name='vgg_std',
                tensor=torch.tensor(
                    [[[0.229]], [[0.224]], [[0.225]]], requires_grad=False).to('cuda')
            )

    def forward(self, x, y):
        if hasattr(self, 'vgg_model'):
            assert x.shape[1] == 3 and y.shape[1] == 3,\
                'VGG model takes 3 chennel images.'

            # normalization
            x = x.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
            y = y.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())

            # picking up vgg feature maps
            x = getattr(self.vgg_model(x), self.vgg_layer)
            y = getattr(self.vgg_model(y), self.vgg_layer)

        return F.contextual_loss(x, y, self.band_width, self.loss_type)