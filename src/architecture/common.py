import math
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from ..utils.helper_functions import exists

# Exponential Moving Average (EMA) class
class EMA:
    def __init__(self, beta: float) -> None:
        """
        Initialize the EMA object.
        :param beta: Smoothing factor for exponential moving average (0 < beta < 1).
        """
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model: nn.Module, current_model: nn.Module) -> None:
        """
        Update the moving average model parameters using the current model parameters.
        :param ma_model: The model storing the moving average parameters.
        :param current_model: The current model with the latest parameters.
        """
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old: Optional[Tensor], new: Tensor) -> Tensor:
        """
        Compute the updated moving average.
        :param old: Previous moving average value.
        :param new: Current value.
        :return: Updated average value.
        """
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

# Residual connection wrapper
class Residual(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        """
        Initialize the Residual wrapper.
        :param fn: Function or module whose output is added to the input.
        """
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Forward pass with residual connection.
        :param x: Input tensor.
        :return: Output tensor with residual addition.
        """
        return self.fn(x, *args, **kwargs) + x

# Sinusoidal positional embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        """
        Initialize sinusoidal positional embedding.
        :param dim: Embedding dimension.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass to compute sinusoidal positional embeddings.
        :param x: Input tensor of shape [batch, 1].
        :return: Positional embeddings of shape [batch, dim].
        """
        device = x.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = x[:, None] * emb[None, :]  # Compute sin and cos components
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # Concatenate sin and cos
        return emb

# 3D Upsampling layer
def Upsample(dim: int) -> nn.ConvTranspose3d:
    """
    Define a 3D transposed convolution for upsampling.
    :param dim: Input and output channel dimension.
    :return: 3D transposed convolutional layer.
    """
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

# 3D Downsampling layer
def Downsample(dim: int) -> nn.Conv3d:
    """
    Define a 3D convolution for downsampling.
    :param dim: Input and output channel dimension.
    :return: 3D convolutional layer.
    """
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

# Custom LayerNorm for 3D tensors
class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        """
        Initialize a custom LayerNorm for 3D tensors.
        :param dim: Channel dimension.
        :param eps: Small value to prevent division by zero.
        """
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))  # Scale parameter

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for LayerNorm.
        :param x: Input tensor of shape [batch, channels, depth, height, width].
        :return: Normalized tensor.
        """
        # Compute mean and variance across the channel dimension
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma

# Root Mean Square (RMS) Norm
class RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        """
        Initialize RMSNorm.
        :param dim: Channel dimension.
        """
        super().__init__()
        self.scale = dim ** 0.5  # Scaling factor
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1, 1))  # Scale parameter

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for RMSNorm.
        :param x: Input tensor of shape [batch, channels, depth, height, width].
        :return: Normalized tensor.
        """
        return F.normalize(x, dim=1) * self.scale * self.gamma

# Pre-normalization wrapper
class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        """
        Initialize a pre-normalization wrapper.
        :param dim: Channel dimension.
        :param fn: Function or module to apply after normalization.
        """
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Forward pass with pre-normalization.
        :param x: Input tensor.
        :return: Output tensor after normalization and function application.
        """
        x = self.norm(x)  # Apply LayerNorm
        return self.fn(x, **kwargs)  # Apply the function
