import torch
from torch.utils import data
from torchvision import transforms as T
from pathlib import Path
from functools import partial
from typing import Optional, Tuple, List
from src.data.utils import gif_to_tensor, cast_num_frames, identity
from src.utils.helper_functions import exists


class Dataset(data.Dataset):
    """
    A PyTorch Dataset class for loading GIF files as tensors with optional text annotations.

    Attributes:
        folder (str): Path to the folder containing the dataset.
        image_size (int): Desired size of the images (both width and height).
        channels (int): Number of channels in the images (default: 3 for RGB).
        num_frames (int): Number of frames to retain in each GIF.
        horizontal_flip (bool): Whether to apply random horizontal flips to the images.
        force_num_frames (bool): Whether to enforce a fixed number of frames per sample.
        exts (List[str]): List of file extensions to include (default: ['gif']).
    """
    def __init__(
        self,
        folder: str,
        image_size: int,
        channels: int = 3,
        num_frames: int = 16,
        horizontal_flip: bool = False,
        force_num_frames: bool = True,
        exts: List[str] = ['gif']
    ) -> None:
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels

        # Collect all valid file paths with the specified extensions
        self.paths = [
            p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')
        ]

        # Function to adjust the number of frames (if required)
        self.cast_num_frames_fn = partial(cast_num_frames, frames=num_frames) if force_num_frames else identity

        # Define the image transformation pipeline
        self.transform = T.Compose([
            T.Resize(image_size),  # Resize to the desired size
            T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),  # Optional horizontal flip
            T.CenterCrop(image_size),  # Crop the image at the center
            T.ToTensor()  # Convert the image to a tensor
        ])

    def __len__(self) -> int:
        """
        Returns the number of items in the dataset.
        """
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Optional[str]]:
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, Optional[str]]: A tuple containing the image tensor and the associated text (if available).
        """
        path = self.paths[index]

        # Load the GIF as a tensor and apply transformations
        tensor = gif_to_tensor(path, self.channels, transform=self.transform)

        # Adjust the number of frames if necessary
        tensor = self.cast_num_frames_fn(tensor)

        # Check for a corresponding text file and load its contents (if it exists)
        text_path = path.with_suffix(".txt")
        if text_path.exists():
            with open(text_path, 'r') as f:
                text = f.read()
                return tensor, text
        else:
            return tensor, None
