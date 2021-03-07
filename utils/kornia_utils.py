from typing import Callable, Tuple, Union, List, Optional, Dict, cast
import torch
from kornia.augmentation import AugmentationBase2D
from kornia.filters import GaussianBlur2d

class GaussianBlur(AugmentationBase2D):

    r"""Apply gaussian blur given tensor image or a batch of tensor images randomly.
    Args:
        kernel_size (Tuple[int, int]): the size of the kernel.
        sigma (Tuple[float, float]): the standard deviation of the kernel.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
            input tensor. If ``False`` and the input is a tuple the applied transformation wont be concatenated.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        p (float): probability of applying the transformation. Default value is 0.5.
    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`
    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 3, 3)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.
    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 5, 5)
        >>> blur = GaussianBlur((3, 3), (0.1, 2.0), p=1.)
        >>> blur(input)
        tensor([[[[0.6699, 0.4645, 0.3193, 0.1741, 0.1955],
                  [0.5422, 0.6657, 0.6261, 0.6527, 0.5195],
                  [0.3826, 0.2638, 0.1902, 0.1620, 0.2141],
                  [0.6329, 0.6732, 0.5634, 0.4037, 0.2049],
                  [0.8307, 0.6753, 0.7147, 0.5768, 0.7097]]]])
    """

    def __init__(self, kernel_size: Tuple[int, int],
                 sigma: Tuple[float, float],
                 border_type: str = 'reflect',
                 return_transform: bool = False,
                 same_on_batch: bool = False,
                 p: float = 0.5) -> None:
        super(GaussianBlur, self).__init__(
            p=p, return_transform=return_transform, same_on_batch=same_on_batch, p_batch=1.)
        self.transform = GaussianBlur2d(kernel_size, sigma, border_type)

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"({super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return dict()

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.transform(input)