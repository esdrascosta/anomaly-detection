import torch as th
# -*- coding: utf-8 -*-
import torch
from torch.nn.modules.loss import _Loss as Loss
import torchvision
from torch.nn import functional as F
from losses.ssim import SSIM

def consine(p, z):
    return F.cosine_similarity(p, z, dim=-1).mean()

def bce_loss(recon_x, x):
    return F.binary_cross_entropy_with_logits(recon_x, x, reduction='mean')


class PerceptualLoss(Loss):

    def __init__(self, ssim=False, lamb=1, inner_loss='cos', input_channels=1):
        super(PerceptualLoss, self).__init__()
        self.ssim = ssim
        self.lamb = lamb
        self.inner_loss = inner_loss
        self.input_channels = input_channels

        if inner_loss == 'mse':
            self.loss_fn = F.mse_loss
        elif inner_loss == 'cos':
            self.loss_fn = consine
        else:
            self.loss_fn = F.l1_loss

        if ssim:
            self.ssim_loss = SSIM()

        vgg16 = torchvision.models.vgg16(pretrained=True)
        blocks = [
            vgg16.features[:4].eval(),
            vgg16.features[4:9].eval(),
            vgg16.features[9:16].eval(),
            vgg16.features[16:23].eval()
        ]
        # for bl in blocks:
        #     for p in bl:
        #         p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, rec_x, x):
        loss = 0.0

        bt, ch, h, w = x.size()
        if self.input_channels == 1:
            rec_x = rec_x.expand(bt, 3, h, w)
            x = x.expand(bt, 3, h, w)

        for block in self.blocks:
            block = block.to(rec_x.device)
            rec_x = block(rec_x)
            x = block(x)
            loss += self.loss_fn(rec_x, x)
        if self.ssim:
            return loss + (self.lamb * (1 - self.ssim_loss(rec_x, x)))
        else:
            return loss

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            bl.to('cuda')
            for p in bl.parameters():
                p.requires_grad = False
            
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1,3,1,1))

        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss