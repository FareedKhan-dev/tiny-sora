import torch
from torch import nn, einsum
from torch import Tensor
from typing import Callable, Optional
from einops import rearrange
from ..utils.helper_functions import exists
from einops_exts import rearrange_many

# A helper class to rearrange tensors and apply a function
class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops: str, to_einops: str, fn: Callable[[Tensor], Tensor]) -> None:
        """
        Initialize with einops patterns for rearranging data and a function to apply between rearrangements.
        :param from_einops: Einops pattern for the original arrangement of the tensor.
        :param to_einops: Einops pattern for rearranging the tensor.
        :param fn: Function to apply to the rearranged tensor.
        """
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Rearranges the tensor, applies the function, and then restores the original arrangement.
        :param x: Input tensor.
        :param kwargs: Additional arguments for the function.
        :return: Tensor with transformations applied.
        """
        shape = x.shape
        # Extract dimensions from the original tensor shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        # Rearrange from the original to the target pattern
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        # Apply the provided function
        x = self.fn(x, **kwargs)
        # Rearrange back to the original pattern
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x

# Implementation of a scaled multi-head self-attention module
class Attention(nn.Module):
    def __init__(
        self,
        dim: int,           # Input dimensionality
        heads: int = 4,     # Number of attention heads
        dim_head: int = 32, # Dimensionality of each attention head
        rotary_emb: Optional[nn.Module] = None # Optional rotary embeddings for positional encoding
    ) -> None:
        """
        Initialize the Attention module.
        :param dim: Dimensionality of the input features.
        :param heads: Number of attention heads.
        :param dim_head: Dimensionality of each head.
        :param rotary_emb: Optional rotary embedding module for positional encoding.
        """
        super().__init__()
        self.scale = dim_head ** -0.5  # Scale factor for dot-product attention
        self.heads = heads
        hidden_dim = dim_head * heads  # Total hidden dimension across all heads

        self.rotary_emb = rotary_emb  # Rotary embeddings for better positional encoding
        # Linear layer to compute queries, keys, and values
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        # Linear layer for the output
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
        self,
        x: Tensor,                 # Input tensor of shape [batch, seq_len, dim]
        pos_bias: Optional[Tensor] = None,     # Optional positional bias tensor
        focus_present_mask: Optional[Tensor] = None # Mask to focus attention on the current position
    ) -> Tensor:
        """
        Forward pass of the Attention module.
        :param x: Input tensor of shape [batch, seq_len, dim].
        :param pos_bias: Optional positional bias to add to attention scores.
        :param focus_present_mask: Mask to restrict attention to the current position.
        :return: Attention output tensor.
        """
        n, device = x.shape[-2], x.device  # Sequence length and device

        # Compute queries, keys, and values
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # If focusing only on the current position, return values directly
        if exists(focus_present_mask) and focus_present_mask.all():
            values = qkv[-1]
            return self.to_out(values)

        # Rearrange queries, keys, and values for multi-head attention
        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)

        # Scale the queries
        q = q * self.scale

        # Apply rotary embeddings to queries and keys, if present
        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # Compute scaled dot-product attention scores
        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # Add positional bias if provided
        if pos_bias is not None:
            sim = sim + pos_bias

        # Handle focus present mask if not all tokens are attended to
        if focus_present_mask is not None and not (~focus_present_mask).all():
            # Full attention and self-attention masks
            attend_all_mask = torch.ones((n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            # Create the final mask based on the focus_present_mask
            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            # Apply the mask to the similarity scores
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # Stabilize softmax by subtracting the maximum score
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        # Compute attention probabilities
        attn = sim.softmax(dim=-1)

        # Compute attention output by weighted sum of values
        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        # Rearrange output back to original format
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)
