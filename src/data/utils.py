from PIL import Image
import torch
from torchvision import transforms as T
from ..utils.helper_functions import exists

# Mapping of channel numbers to PIL image modes
CHANNELS_TO_MODE = {
    1: 'L',       # Grayscale (1 channel)
    3: 'RGB',     # RGB (3 channels)
    4: 'RGBA'     # RGBA (4 channels)
}

def seek_all_images(img: Image.Image, channels: int = 3):
    """
    Iterates through all frames of a GIF and converts them to a specified mode.

    Args:
        img (Image.Image): The PIL image object (GIF).
        channels (int): Number of channels (1, 3, or 4).

    Yields:
        Image.Image: A single frame converted to the specified mode.
    """
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)  # Move to the i-th frame
            yield img.convert(mode)  # Convert to the specified mode
        except EOFError:
            break  # End of GIF
        i += 1

def video_tensor_to_gif(
    tensor: torch.Tensor,
    path: str,
    duration: int = 120,
    loop: int = 0,
    optimize: bool = True
):
    """
    Converts a video tensor to a GIF and saves it to a file.

    Args:
        tensor (torch.Tensor): The video tensor of shape (channels, frames, height, width).
        path (str): Path to save the GIF file.
        duration (int): Duration (in ms) of each frame.
        loop (int): Number of times the GIF should loop. 0 means infinite loop.
        optimize (bool): Whether to optimize the GIF for smaller size.

    Returns:
        List[Image.Image]: The list of frames (PIL images) saved to the GIF.
    """
    images = map(T.ToPILImage(), tensor.unbind(dim=1))  # Convert each frame to PIL Image
    first_img, *rest_imgs = images
    first_img.save(
        path,
        save_all=True,
        append_images=rest_imgs,
        duration=duration,
        loop=loop,
        optimize=optimize
    )
    return images

def gif_to_tensor(
    path: str,
    channels: int = 3,
    transform: T.Compose = T.ToTensor()
) -> torch.Tensor:
    """
    Converts a GIF to a video tensor.

    Args:
        path (str): Path to the GIF file.
        channels (int): Number of channels (1, 3, or 4).
        transform (T.Compose): Transformation to apply to each frame (default: T.ToTensor).

    Returns:
        torch.Tensor: The video tensor of shape (channels, frames, height, width).
    """
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels=channels)))  # Transform each frame
    return torch.stack(tensors, dim=1)  # Stack frames along the temporal dimension

def identity(t, *args, **kwargs):
    """
    Identity function that returns the input unchanged.
    Useful as a placeholder for optional operations.
    
    Args:
        t: Input tensor.

    Returns:
        The input tensor unchanged.
    """
    return t

def normalize_img(t: torch.Tensor) -> torch.Tensor:
    """
    Normalizes an image tensor to the range [-1, 1].

    Args:
        t (torch.Tensor): Input image tensor in the range [0, 1].

    Returns:
        torch.Tensor: Normalized image tensor in the range [-1, 1].
    """
    return t * 2 - 1

def unnormalize_img(t: torch.Tensor) -> torch.Tensor:
    """
    Unnormalizes an image tensor to the range [0, 1].

    Args:
        t (torch.Tensor): Input image tensor in the range [-1, 1].

    Returns:
        torch.Tensor: Unnormalized image tensor in the range [0, 1].
    """
    return (t + 1) * 0.5

def cast_num_frames(t: torch.Tensor, *, frames: int) -> torch.Tensor:
    """
    Ensures a tensor has a specific number of frames by trimming or padding.

    Args:
        t (torch.Tensor): Input tensor of shape (channels, frames, height, width).
        frames (int): Desired number of frames.

    Returns:
        torch.Tensor: Tensor with the specified number of frames.
    """
    f = t.shape[1]  # Current number of frames

    if f == frames:
        return t  # No change if the number of frames matches

    if f > frames:
        return t[:, :frames]  # Trim extra frames

    # Pad with zeros if there are fewer frames
    return torch.nn.functional.pad(t, (0, 0, 0, 0, 0, frames - f))
