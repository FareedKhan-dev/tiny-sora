import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from typing import Optional, Tuple
from src.utils.helper_functions import exists
from src.architecture.common import RMSNorm
from einops import rearrange
from einops_exts import rearrange_many

# Basic block used in ResNetBlock
class Block(nn.Module):
    def __init__(self, dim: int, dim_out: int) -> None:
        """
        Initialize a basic convolutional block.
        :param dim: Input channel dimension.
        :param dim_out: Output channel dimension.
        """
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x: Tensor, scale_shift: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
        """
        Forward pass for the block.
        :param x: Input tensor of shape [batch, channels, depth, height, width].
        :param scale_shift: Optional tuple of (scale, shift) tensors for modulation.
        :return: Processed tensor.
        """
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)

# ResNet-inspired block
class ResnetBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, *, time_emb_dim: Optional[int] = None) -> None:
        """
        Initialize a ResNet block.
        :param dim: Input channel dimension.
        :param dim_out: Output channel dimension.
        :param time_emb_dim: Optional dimension for time embeddings.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: Tensor, time_emb: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass for the ResNet block.
        :param x: Input tensor of shape [batch, channels, depth, height, width].
        :param time_emb: Optional time embedding tensor of shape [batch, time_emb_dim].
        :return: Output tensor.
        """
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time_emb must be passed in when time_emb_dim is defined'
            # Process time embedding through MLP
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        # Apply the two convolutional blocks with optional scale/shift
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

# Spatial linear attention block
class SpatialLinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32) -> None:
        """
        Initialize the spatial linear attention module.
        :param dim: Input channel dimension.
        :param heads: Number of attention heads.
        :param dim_head: Dimension of each attention head.
        """
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        # Layers for query, key, value computation and output projection
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for spatial linear attention.
        :param x: Input tensor of shape [batch, channels, depth, height, width].
        :return: Output tensor of the same shape as input.
        """
        b, c, f, h, w = x.shape  # Batch size, channels, depth, height, width
        # Combine batch and depth dimensions for processing
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        # Compute queries, keys, and values
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(qkv, 'b (h c) x y -> b h c (x y)', h=self.heads)

        # Apply softmax normalization to queries and keys
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        # Scale the queries
        q = q * self.scale

        # Compute the context by multiplying keys and values
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        # Compute the output by attending to the context with the queries
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)

        # Project the output back to the original dimension
        out = self.to_out(out)
        # Restore the original batch and depth dimensions
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)