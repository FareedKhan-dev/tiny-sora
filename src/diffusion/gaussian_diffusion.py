import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from ..utils.helper_functions import default, exists, prob_mask_like, is_list_str
from ..text.text_handler import bert_embed, tokenize
from einops_exts import check_shape
from tqdm import tqdm

def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """
    Extract values from a tensor based on the given timesteps.
    
    Args:
        a (torch.Tensor): The tensor from which to gather values.
        t (torch.Tensor): The timestep indices.
        x_shape (torch.Size): Shape of the tensor to return.

    Returns:
        torch.Tensor: A tensor reshaped according to x_shape.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule for generating betas as proposed in 
    "Improved Techniques for Training Score-based Generative Models" (https://openreview.net/forum?id=-NEXDKk8gZ).
    
    Args:
        timesteps (int): The number of timesteps.
        s (float, optional): A scaling factor. Default is 0.008.

    Returns:
        torch.Tensor: A tensor of betas for each timestep.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)

class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion model for score-based generative models.
    
    Args:
        denoise_fn (nn.Module): Denoising function (neural network) to predict the noise.
        image_size (int): The size of the image (height/width).
        num_frames (int): The number of frames for videos.
        text_use_bert_cls (bool, optional): Whether to use BERT CLS embeddings for conditioning.
        channels (int, optional): Number of image channels (default is 3 for RGB images).
        timesteps (int, optional): Number of timesteps for diffusion (default is 1000).
        loss_type (str, optional): Type of loss function ('l1' or 'l2').
        use_dynamic_thres (bool, optional): Whether to use dynamic thresholding during sampling.
        dynamic_thres_percentile (float, optional): Percentile for dynamic thresholding (default is 0.9).
    """
    def __init__(
        self,
        denoise_fn: nn.Module,
        *,
        image_size: int,
        num_frames: int,
        text_use_bert_cls: bool = False,
        channels: int = 3,
        timesteps: int = 1000,
        loss_type: str = 'l1',
        use_dynamic_thres: bool = False,  # from the Imagen paper
        dynamic_thres_percentile: float = 0.9
    ) -> None:
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # register buffer helper function that casts float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # text conditioning parameters
        self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling
        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_mean_variance(self, x_start: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the mean, variance, and log variance for the forward diffusion process.
        
        Args:
            x_start (torch.Tensor): The starting data (image/video) at timestep 0.
            t (torch.Tensor): The timesteps.

        Returns:
            tuple: A tuple containing the mean, variance, and log variance.
        """
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Predict the starting image from the noisy image using the reverse process.
        
        Args:
            x_t (torch.Tensor): Noisy image at timestep t.
            t (torch.Tensor): The current timestep.
            noise (torch.Tensor): The noise that has been added.

        Returns:
            torch.Tensor: The predicted starting image (x_0).
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the posterior distribution q(x_{t-1} | x_t, x_0) for the reverse process.
        
        Args:
            x_start (torch.Tensor): The starting image (x_0).
            x_t (torch.Tensor): The noisy image at timestep t.
            t (torch.Tensor): The current timestep.

        Returns:
            tuple: A tuple containing the posterior mean, variance, and log variance.
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x: torch.Tensor, t: torch.Tensor, clip_denoised: bool, cond: torch.Tensor = None, cond_scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the mean, variance, and log variance for the reverse process.
        
        Args:
            x (torch.Tensor): The current noisy image.
            t (torch.Tensor): The current timestep.
            clip_denoised (bool): Whether to clip the denoised image.
            cond (torch.Tensor, optional): Condition tensor (e.g., text embeddings).
            cond_scale (float, optional): Scaling factor for conditioning.

        Returns:
            tuple: A tuple containing the model mean, variance, and log variance.
        """
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn.forward_with_cond_scale(x, t, cond=cond, cond_scale=cond_scale))

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )

                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor = None, cond_scale: float = 1.0, clip_denoised: bool = True) -> torch.Tensor:
        """
        Sample a new image from the model given a noisy image and timestep.
        
        Args:
            x (torch.Tensor): The noisy image.
            t (torch.Tensor): The current timestep.
            cond (torch.Tensor, optional): The conditioning tensor (e.g., text embeddings).
            cond_scale (float, optional): Scaling factor for conditioning.
            clip_denoised (bool, optional): Whether to clip the denoised image.

        Returns:
            torch.Tensor: The sampled image at timestep t.
        """
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, cond=cond, cond_scale=cond_scale)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, shape: torch.Size, cond: torch.Tensor = None, cond_scale: float = 1.0) -> torch.Tensor:
        """
        Perform a loop of sampling steps starting from random noise.
        
        Args:
            shape (torch.Size): The shape of the generated image.
            cond (torch.Tensor, optional): The conditioning tensor (e.g., text embeddings).
            cond_scale (float, optional): Scaling factor for conditioning.

        Returns:
            torch.Tensor: The final generated image.
        """
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), cond=cond, cond_scale=cond_scale)

        return (img + 1) * 0.5

    @torch.inference_mode()
    def sample(self, cond: torch.Tensor = None, cond_scale: float = 1.0, batch_size: int = 16) -> torch.Tensor:
        """
        Generate a sample given some condition (e.g., text or latent).

        Args:
            cond (torch.Tensor, optional): The conditioning tensor (e.g., text embeddings).
            cond_scale (float, optional): Scaling factor for conditioning.
            batch_size (int, optional): Batch size for sampling.

        Returns:
            torch.Tensor: The generated sample.
        """
        device = next(self.denoise_fn.parameters()).device

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond)).to(device)

        batch_size = cond.shape[0] if exists(cond) else batch_size
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames
        return self.p_sample_loop((batch_size, channels, num_frames, image_size, image_size), cond=cond, cond_scale=cond_scale)

    @torch.inference_mode()
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, t: torch.Tensor = None, lam: float = 0.5) -> torch.Tensor:
        """
        Interpolate between two images given a blending coefficient lam.
        
        Args:
            x1 (torch.Tensor): The first image tensor.
            x2 (torch.Tensor): The second image tensor.
            t (torch.Tensor, optional): The timestep to interpolate at.
            lam (float, optional): The blending coefficient between the two images.

        Returns:
            torch.Tensor: The interpolated image.
        """
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """
        Sample from the forward diffusion process given an image and noise.
        
        Args:
            x_start (torch.Tensor): The starting image tensor.
            t (torch.Tensor): The current timestep.
            noise (torch.Tensor, optional): The noise to be added.

        Returns:
            torch.Tensor: The noisy image.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, cond: torch.Tensor = None, noise: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Compute the loss for a single timestep in the diffusion process.
        
        Args:
            x_start (torch.Tensor): The starting image (x_0).
            t (torch.Tensor): The timestep.
            cond (torch.Tensor, optional): The conditioning tensor (e.g., text embeddings).
            noise (torch.Tensor, optional): The noise tensor.
        
        Returns:
            torch.Tensor: The computed loss value.
        """
        b, c, f, h, w, device = *x_start.shape, x_start.device
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond), return_cls_repr=self.text_use_bert_cls)
            cond = cond.to(device)

        x_recon = self.denoise_fn(x_noisy, t, cond=cond, **kwargs)

        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        The forward pass of the diffusion model.
        
        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The loss for the diffusion process.
        """
        b, device, img_size = x.shape[0], x.device, self.image_size
        check_shape(x, 'b c f h w', c=self.channels, f=self.num_frames, h=img_size, w=img_size)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        x = (x * 2) - 1
        return self.p_losses(x, t, *args, **kwargs)
