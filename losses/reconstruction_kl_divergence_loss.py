# -*- coding: utf-8 -*-
import torch
from torch.nn.modules.loss import _Loss as Loss
from torch.nn import functional as F


class ReconsKLDivergenceLoss(Loss):
    def __init__(self, beta=1):
        super(ReconsKLDivergenceLoss, self).__init__()

        self.beta = beta

    def forward(self, recon_x, x, mu, logvar):
        
        # BCE = F.mse_loss(recon_x, x, reduction='sum')
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + (self.beta * KLD)
