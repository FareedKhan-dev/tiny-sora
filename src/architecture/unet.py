import torch
from torch import nn
from functools import partial
from typing import List, Optional
from rotary_embedding_torch import RotaryEmbedding
from src.architecture.attention import Attention, EinopsToAndFrom
from src.architecture.blocks import ResnetBlock, SpatialLinearAttention
from src.architecture.common import PreNorm, Downsample, Upsample, SinusoidalPosEmb
from src.utils.helper_functions import default, exists, is_odd, prob_mask_like
from src.architecture.relative_position_bias import RelativePositionBias
from src.text.text_handler import BERT_MODEL_DIM
from einops import rearrange

# Residual block with optional text conditioning
class Residual(nn.Module):
    """
    A residual block with optional text conditioning.

    Attributes:
        fn (nn.Module): The main function of the block.
    """
    
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    # Forward pass with optional text conditioning
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class Unet3D(nn.Module):
    """
    A 3D U-Net model with attention and optional text conditioning.
    Designed for video or volumetric data, incorporating temporal, spatial, and text-based attention.

    Attributes:
        dim (int): Base dimensionality for the model.
        cond_dim (Optional[int]): Dimensionality of conditioning input.
        out_dim (Optional[int]): Dimensionality of the output.
        dim_mults (List[int]): Multipliers for feature map dimensionality across layers.
        channels (int): Number of input channels.
        attn_heads (int): Number of attention heads.
        attn_dim_head (int): Dimensionality of each attention head.
        use_bert_text_cond (bool): Whether to use BERT for text conditioning.
        init_dim (Optional[int]): Initial dimensionality of the model.
        init_kernel_size (int): Kernel size for the initial convolution.
        use_sparse_linear_attn (bool): Whether to use sparse linear attention.
        block_type (str): Type of residual block to use.
    """
    def __init__(
        self,
        dim: int,
        cond_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        dim_mults: List[int] = (1, 2, 4, 8),
        channels: int = 3,
        attn_heads: int = 8,
        attn_dim_head: int = 32,
        use_bert_text_cond: bool = True,
        init_dim: Optional[int] = None,
        init_kernel_size: int = 7,
        use_sparse_linear_attn: bool = True,
        block_type: str = 'resnet'
    ) -> None:
        super().__init__()
        self.channels = channels

        # Temporal attention with rotary embeddings
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        temporal_attn = lambda dim: EinopsToAndFrom(
            'b c f h w', 'b (h w) f c',
            Attention(dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb)
        )

        # Relative positional bias for temporal attention
        self.time_rel_pos_bias = RelativePositionBias(heads=attn_heads, max_distance=32)

        # Initial convolution
        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size), "Initial kernel size must be odd."
        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(
            channels, init_dim,
            (1, init_kernel_size, init_kernel_size),
            padding=(0, init_padding, init_padding)
        )

        self.init_temporal_attn = Residual(PreNorm(init_dim, temporal_attn(init_dim)))

        # Calculate dimensions for each resolution
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Time embedding
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # Text conditioning
        self.has_cond = exists(cond_dim) or use_bert_text_cond
        cond_dim = BERT_MODEL_DIM if use_bert_text_cond else cond_dim
        self.null_cond_emb = nn.Parameter(torch.randn(1, cond_dim)) if self.has_cond else None
        cond_dim = time_dim + int(cond_dim or 0)

        # Downsampling layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        # Residual block selection
        block_klass = ResnetBlock
        block_klass_cond = partial(block_klass, time_emb_dim=cond_dim)

        # Create downsampling layers
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(dim_out, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        # Middle layers
        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)
        spatial_attn = EinopsToAndFrom('b c f h w', 'b f (h w) c', Attention(mid_dim, heads=attn_heads))
        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))
        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        # Create upsampling layers
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(dim_in, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        # Final layers
        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale: float = 2.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with conditional scaling for classifier-free guidance.
        """
        logits = self.forward(*args, null_cond_prob=0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        null_cond_prob: float = 0.,
        focus_present_mask: Optional[torch.Tensor] = None,
        prob_focus_present: float = 0.0
    ) -> torch.Tensor:
        """
        Main forward pass.
        """
        assert not (self.has_cond and not exists(cond)), 'Conditioning must be provided when cond_dim is specified.'
        batch, device = x.shape[0], x.device
        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like((batch,), prob_focus_present, device=device))

        # Positional bias
        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)

        # Initial convolution and temporal attention
        x = self.init_conv(x)
        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)
        r = x.clone()

        # Time embedding
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # Classifier-free guidance
        if self.has_cond:
            mask = prob_mask_like((batch,), null_cond_prob, device=device)
            cond = torch.where(rearrange(mask, 'b -> b 1'), self.null_cond_emb, cond)
            t = torch.cat((t, cond), dim=-1)

        # Downsampling
        h = []
        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
            h.append(x)
            x = downsample(x)

        # Middle layers
        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
        x = self.mid_block2(x, t)

        # Upsampling
        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
            x = upsample(x)

        # Final layers
        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)
