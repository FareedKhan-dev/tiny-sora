import math
import torch
from torch import nn
from torch import Tensor
from typing import Optional
from einops import rearrange
from ..utils.helper_functions import exists

class RelativePositionBias(nn.Module):
    """
    A module to compute relative position biases for attention mechanisms.
    This is useful in transformer-based architectures to introduce positional information 
    in a relative manner.

    Attributes:
        heads (int): Number of attention heads.
        num_buckets (int): Number of relative position buckets.
        max_distance (int): Maximum relative distance to bucketize.
        relative_attention_bias (nn.Embedding): Embedding layer to learn bias for each bucket.
    """
    def __init__(
        self,
        heads: int = 8,
        num_buckets: int = 32,
        max_distance: int = 128
    ) -> None:
        """
        Initialize the RelativePositionBias module.
        
        Args:
            heads (int): Number of attention heads.
            num_buckets (int): Number of relative position buckets.
            max_distance (int): Maximum relative distance to bucketize.
        """
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position: Tensor, 
        num_buckets: int = 32, 
        max_distance: int = 128
    ) -> Tensor:
        """
        Assigns a relative position to a bucket.

        Args:
            relative_position (Tensor): Tensor of relative positions.
            num_buckets (int): Number of buckets to use.
            max_distance (int): Maximum relative distance to bucketize.
        
        Returns:
            Tensor: Tensor of bucket indices corresponding to the input relative positions.
        """
        ret = 0  # Initialize bucket indices
        n = -relative_position  # Convert to positive relative positions for bucketing

        num_buckets //= 2  # Split buckets into positive and negative
        ret += (n < 0).long() * num_buckets  # Assign negative positions to separate buckets
        n = torch.abs(n)  # Convert to absolute values

        max_exact = num_buckets // 2  # Exact buckets for small distances
        is_small = n < max_exact  # Check if positions fall into exact range

        # Compute bucket indices for larger distances
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))  # Cap indices

        ret += torch.where(is_small, n, val_if_large)  # Combine small and large buckets
        return ret

    def forward(self, n: int, device: torch.device) -> Tensor:
        """
        Compute relative position bias values for a sequence of length `n`.

        Args:
            n (int): Sequence length.
            device (torch.device): Device for computation.
        
        Returns:
            Tensor: Relative position bias tensor of shape [heads, n, n].
        """
        # Compute query and key positions
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        
        # Compute relative positions
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        
        # Bucketize relative positions
        rp_bucket = self._relative_position_bucket(
            rel_pos, 
            num_buckets=self.num_buckets, 
            max_distance=self.max_distance
        )
        
        # Retrieve bias values for each bucket
        values = self.relative_attention_bias(rp_bucket)
        
        # Rearrange to match the attention head dimension
        return rearrange(values, 'i j h -> h i j')
