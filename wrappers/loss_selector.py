from losses.contextual_loss import ContextualLoss
from losses.reconstruction_kl_divergence_loss import ReconsKLDivergenceLoss
import torch
from torch.nn import BCEWithLogitsLoss
from piq import SSIMLoss, GMSDLoss, MultiScaleGMSDLoss

def custom_loss(prediction: torch.Tensor, target: torch.Tensor):
    return MultiScaleGMSDLoss(scale_weights=[1, 1, 1, 1])(prediction, target) + \
            torch.nn.MSELoss()(prediction, target) + \
            SSIMLoss()(prediction, target)

class LossSelector:

    @classmethod
    def select_loss(cls, hparams):

        in_channel = 3 if hparams.dataset == 'cifar-10' or hparams.dataset == 'mvtech-ad' else 1

        if hparams.loss == 'contextual_l1':
            return ContextualLoss(loss='l1', backbone=hparams.loss_backbone, input_channels=in_channel)
        elif hparams.loss == 'contextual_l1_pre_l1':
            return ContextualLoss(loss='l1', add_pre_l1=True, backbone=hparams.loss_backbone, input_channels=in_channel)
        elif hparams.loss == 'contextual_mse':
            return ContextualLoss(loss='mse', backbone=hparams.loss_backbone, input_channels=in_channel)
        elif hparams.loss == 'contextual_bce':
            return ContextualLoss(loss='bce', backbone=hparams.loss_backbone, input_channels=in_channel)
        elif hparams.loss == 'contextual_cos':
            return ContextualLoss(loss='cos', backbone=hparams.loss_backbone, input_channels=in_channel)
        elif hparams.loss == 'l2':
            return torch.nn.MSELoss()
        elif hparams.loss == 'l1':
            return torch.nn.L1Loss()
        elif hparams.loss == 'kl':
            return ReconsKLDivergenceLoss()
        elif hparams.loss == 'msgmsd':
            return MultiScaleGMSDLoss()
        elif hparams.loss == 'custom':
            return custom_loss
        elif hparams.loss == 'BCEWithLogitsLoss':
            return BCEWithLogitsLoss()
        else:
            return ContextualLoss(backbone=hparams.loss_backbone, input_channels=in_channel)
