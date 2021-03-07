import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.filters.median import median_blur
import matplotlib.pyplot as plt

def rgb2yiq(x: torch.Tensor) -> torch.Tensor:
    r"""Convert a batch of RGB images to a batch of YIQ images

    Args:
        x: Batch of images with shape (N, 3, H, W). RGB colour space.

    Returns:
        Batch of images with shape (N, 3, H, W). YIQ colour space.
    """
    yiq_weights = torch.tensor([
        [0.299, 0.587, 0.114],
        [0.5959, -0.2746, -0.3213],
        [0.2115, -0.5227, 0.3112]]).t().to(x)
    x_yiq = torch.matmul(x.permute(0, 2, 3, 1), yiq_weights).permute(0, 3, 1, 2)
    return x_yiq

def gradient_map(x: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
    r""" Compute gradient map for a given tensor and stack of kernels.

    Args:
        x: Tensor with shape (N, C, H, W).
        kernels: Stack of tensors for gradient computation with shape (k_N, k_H, k_W)
    Returns:
        Gradients of x per-channel with shape (N, C, H, W)
    """
    padding = kernels.size(-1) // 2
    grads = torch.nn.functional.conv2d(x, kernels.to(x), padding=padding)

    return torch.sqrt(torch.sum(grads ** 2, dim=-3, keepdim=True))

def prewitt_filter() -> torch.Tensor:
    r"""Utility function that returns a normalized 3x3 Prewitt kernel in X direction
    Returns:
        kernel: Tensor with shape (1, 3, 3)"""
    return torch.tensor([[[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]]]) / 3


def _gmsd(prediction, target, t):
    
    prediction = median_blur(prediction, (3, 3))
    target = median_blur(target, (3, 3))

    kernels = torch.stack([prewitt_filter(), prewitt_filter().transpose(-1, -2)])
    gr_pred_grad = gradient_map(prediction, kernels)
    gi_trgt_grad = gradient_map(target, kernels)

    return ( 2 * gi_trgt_grad * gr_pred_grad + t) / ( gi_trgt_grad ** 2 + gr_pred_grad ** 2 + t)

def show_debug(gmsd_map):
    g = gmsd_map[0][0].detach().cpu().numpy()
    plt.imshow(g, cmap='jet', interpolation='none')

def msgmsd(prediction, target, t=170):

    prediction = prediction * 255
    target = target * 255

    num_channels = prediction.size(1)
    if num_channels == 3:
        prediction = rgb2yiq(prediction)[:, :1]
        target = rgb2yiq(target)[:, :1]
    num_scales = 4
    img_size = target.size(-1)
    gmsd_maps = []
    for scale in range(num_scales):
        if scale > 0:
            # Average by 2x2 filter and downsample
            up_pad = 0
            down_pad = max(prediction.shape[2] % 2, prediction.shape[3] % 2)
            pad_to_use = [up_pad, down_pad, up_pad, down_pad]
            prediction = F.pad(prediction, pad=pad_to_use)
            target = F.pad(target, pad=pad_to_use)
            prediction = F.avg_pool2d(prediction, kernel_size=2, stride=2, padding=0)
            target = F.avg_pool2d(target, kernel_size=2, stride=2, padding=0)

        gmsd_map = _gmsd(prediction[:, :1], target[:, :1], t=t)
        
        # import pdb; pdb.set_trace()
        # show_debug(gmsd_map)
        if scale > 0:
            gmsd_map = F.interpolate(gmsd_map, size=img_size, mode='bilinear')
        gmsd_maps.append(gmsd_map)
    return 1 - torch.cat(gmsd_maps, dim=1).mean(dim=1, keepdim=True)
        

if __name__ == "__main__":
    prediction = torch.rand((2, 3, 64, 64))
    target = torch.rand((2, 3, 64, 64))
    msgms_map = msgmsd(prediction, target)
    ones = torch.ones(msgms_map.shape).to(msgms_map)
    result =  ones - median_blur(msgms_map, (21, 21))