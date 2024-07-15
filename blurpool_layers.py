import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BlurPool(nn.Module):
    """
    BlurPool module for performing anti-aliasing downsampling described in
    https://arxiv.org/pdf/1904.11486

    Args:
    - in_channels (int): Number of channels in the input tensor.
    - filter_size (int): Size of the filter for pooling.
    - stride (int): Stride of the pooling operation.
    - padding (int): Padding size to apply around the input.
    - padding_mode (str, optional): Padding mode to use. Default is 'reflect'.
    - value (float, optional): Value to fill for padding if padding_mode is 'constant'. Default is None.

    Raises:
    - AssertionError: If padding_mode is 'zeros' and value is not 0 or None.

    Methods:
    - _create_filter(): Placeholder method to create the blur filter.
    - _get_padding(): Placeholder method to calculate padding based on parameters.
    - forward(x): Placeholder method for the forward pass through the module.
    """
    def __init__(self, in_channels: int, filter_size: int, stride: int, padding: int, padding_mode: str = 'reflect', value: Optional[float] = None):
        super(BlurPool, self).__init__()
        self.in_channels = in_channels
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        if padding_mode == 'zeros':
            self.padding_mode = 'constant'
            assert value == 0 or value is None, f"value for {padding_mode} should be 0 or None"
        self.value = value

    def _create_filter(self) -> torch.Tensor:
        raise NotImplementedError
    
    def _get_padding(self) -> list:
        raise NotImplementedError
    
    def forward(self, x) -> torch.Tensor:
        raise NotImplementedError


class BlurPool1d(BlurPool):
    def __init__(self, in_channels, filter_size = 3, stride = 2, padding = 0, padding_mode = 'reflect', value = None):
        super(BlurPool1d, self).__init__(in_channels, filter_size, stride, padding, padding_mode, value)
        self.register_buffer('blurpool_filter', self._create_filter())
        self.pad = self._get_padding()

    def _create_filter(self) -> torch.Tensor:
        coeff = [1]
        for k in range(1, self.filter_size):
            coeff.append(coeff[-1] * (self.filter_size - k) // k)
        coeff_t = torch.tensor(coeff, dtype=torch.float32)
        flt = torch.Tensor(coeff_t)
        flt = flt / torch.sum(flt)
        return flt.view(1, 1, self.filter_size).repeat(self.in_channels, 1, 1)
    
    def _get_padding(self) -> list:
        even = (self.filter_size - 1) // 2 + self.padding
        odd = (self.filter_size) // 2 + self.padding
        pad = [even, odd]
        return pad
        
    def forward(self, x) -> torch.Tensor:
        x = F.pad(x, self.pad, mode=self.padding_mode, value=self.value)
        x = F.conv1d(x, self.blurpool_filter, stride=self.stride, groups=self.in_channels)
        return x


class BlurPool2d(BlurPool):
    def __init__(self, in_channels, filter_size = 4, stride = 2, padding = 0, padding_mode = 'reflect', value = None):
        super(BlurPool2d, self).__init__(in_channels, filter_size, stride, padding, padding_mode, value)
        self.register_buffer('blurpool_filter', self._create_filter())
        self.pad = self._get_padding()

    def _create_filter(self) -> torch.Tensor:
        coeff = [1]
        for k in range(1, self.filter_size):
            coeff.append(coeff[-1] * (self.filter_size - k) // k)
        coeff_t = torch.tensor(coeff, dtype=torch.float32)
        flt = coeff_t[:, None] * coeff_t[None, :]
        flt = flt / torch.sum(flt)
        return flt.view(1, 1, self.filter_size, self.filter_size).repeat(self.in_channels, 1, 1, 1)

    def _get_padding(self) -> list:
        even = (self.filter_size - 1) // 2 + self.padding
        odd = (self.filter_size) // 2 + self.padding
        pad = [even, odd, even, odd]
        return pad
    
    def forward(self, x) -> torch.Tensor:
        x = F.pad(x, self.pad, mode=self.padding_mode, value=self.value)
        x = F.conv2d(x, self.blurpool_filter, stride=self.stride, groups=self.in_channels)
        return x


class BlurPool3d(BlurPool):
    def __init__(self, in_channels, filter_size = 3, stride = 2, padding = 0, padding_mode = 'reflect', value = None):
        super(BlurPool3d, self).__init__(in_channels, filter_size, stride, padding, padding_mode, value)
        self.register_buffer('blurpool_filter', self._create_filter())
        self.pad = self._get_padding()

    def _create_filter(self) -> torch.Tensor:
        coeff = [1]
        for k in range(1, self.filter_size):
            coeff.append(coeff[-1] * (self.filter_size - k) // k)
        coeff_t = torch.tensor(coeff, dtype=torch.float32)
        flt = coeff_t[:, None, None] * coeff_t[None, :, None] * coeff_t[None, None, :]
        flt = flt / torch.sum(flt)
        return flt.view(1, 1, self.filter_size, self.filter_size, self.filter_size).repeat(self.in_channels, 1, 1, 1, 1)
    
    def _get_padding(self) -> list:
        even = (self.filter_size - 1) // 2 + self.padding
        odd = (self.filter_size) // 2 + self.padding
        pad = [even, odd, even, odd, even, odd]
        return pad
    
    def forward(self, x) -> torch.Tensor:
        x = F.pad(x, self.pad, mode=self.padding_mode, value=self.value)
        x = F.conv3d(x, self.blurpool_filter, stride=self.stride, groups=self.in_channels)
        return x