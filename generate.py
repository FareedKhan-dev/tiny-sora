import argparse
import os
import re
import yaml
import torch
import numpy as np
from PIL import Image
from src.architecture.unet import Unet3D
from src.diffusion.gaussian_diffusion import GaussianDiffusion


# Fixed paths
CONFIG_PATH = "configs/default.yaml"
DEFAULT_MODEL_PATH = "./saved_models"
DEFAULT_OUTPUT_DIR = "./results"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate a video using a trained diffusion model and save as a GIF.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the trained model checkpoint (.pt file).")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save the output GIF.")
    parser.add_argument("--text", type=str, required=True, help="Text prompt for video generation.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for video generation.")
    parser.add_argument("--cond_scale", type=float, default=2.0, help="Conditioning scale for diffusion sampling.")
    return parser.parse_args()


def sanitize_filename(text: str) -> str:
    """Generate a safe filename using the first 2-3 words from the text."""
    words = text.strip().split()[:3]  # Get first 2-3 words
    filename = "_".join(words)  # Join with underscores
    filename = re.sub(r"[^a-zA-Z0-9_]", "", filename)  # Remove special characters
    return filename.lower()


def load_config(config_path: str) -> dict:
    """Load the training configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(model_path: str, config: dict) -> GaussianDiffusion:
    """Load the trained diffusion model using the saved checkpoint and training configuration."""
    model = Unet3D(**config["model"])  # Load model architecture

    diffusion = GaussianDiffusion(
        denoise_fn=model,
        **config["diffusion"]
    ).cuda()

    # Find the latest model checkpoint
    if os.path.isdir(model_path):
        checkpoint_files = [f for f in os.listdir(model_path) if f.endswith(".pt")]
        if not checkpoint_files:
            raise FileNotFoundError(f"No model checkpoint found in {model_path}")
        checkpoint_files.sort()  # Sort by name (you may want to sort by timestamp instead)
        model_path = os.path.join(model_path, checkpoint_files[-1])  # Use latest model

    # Load the model checkpoint
    checkpoint = torch.load(model_path)
    diffusion.load_state_dict(checkpoint["ema"])

    return diffusion


def generate_video(diffusion: GaussianDiffusion, text: str, batch_size: int, cond_scale: float) -> torch.Tensor:
    """Generate a video using the trained diffusion model."""
    with torch.no_grad():
        video = diffusion.sample(cond=[text], batch_size=batch_size, cond_scale=cond_scale)
    return video


def save_video_as_gif_pil(video_tensor: torch.Tensor, output_path: str) -> None:
    """Convert a generated video tensor into a GIF and save it using PIL."""
    
    # Move tensor to CPU, remove batch dimension, and convert to NumPy
    video_np = (video_tensor.squeeze(0).permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8)

    # Convert frames to PIL images
    frames = [Image.fromarray(frame) for frame in video_np]

    # Save as an animated GIF
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=100, loop=0)

    print(f"Saved GIF: {output_path}")


def main():
    args = parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load configuration from YAML (always from fixed path)
    config = load_config(CONFIG_PATH)

    # Load trained model
    diffusion_model = load_model(args.model_path, config)

    # Generate video
    generated_video = generate_video(diffusion_model, args.text, args.batch_size, args.cond_scale)

    # Create a filename based on the text prompt
    gif_filename = sanitize_filename(args.text) + ".gif"
    output_path = os.path.join(args.output_dir, gif_filename)

    # Save video as GIF using PIL
    save_video_as_gif_pil(generated_video, output_path)


if __name__ == "__main__":
    main()