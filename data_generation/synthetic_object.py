import os
import random
from typing import Tuple, Dict, List
from PIL import Image, ImageDraw
import numpy as np
import colorsys
from tqdm import tqdm

def create_directory_structure() -> str:
    """
    Creates the necessary directories for storing the dataset.

    Returns:
        str: The training directory path.
    """

    training_dir: str = os.path.join("../training_data")
    os.makedirs(training_dir, exist_ok=True)
    
    return training_dir

def get_random_color() -> Tuple[int, int, int]:
    """
    Generates a random vibrant RGB color tuple.

    Returns:
        Tuple[int, int, int]: A tuple representing an RGB color.
    """
    hue: float = random.random()
    saturation: float = random.uniform(0.5, 1.0)
    value: float = random.uniform(0.5, 1.0)
    rgb: Tuple[float, float, float] = colorsys.hsv_to_rgb(hue, saturation, value)
    return tuple(int(x * 255) for x in rgb)

def generate_random_pattern() -> str:
    """
    Chooses a random background pattern from a predefined set.

    Returns:
        str: The name of the selected pattern.
    """
    patterns: List[str] = ['dots', 'grid', 'stripes', 'solid']
    return random.choice(patterns)

def apply_background_pattern(img: Image.Image, pattern_type: str) -> Tuple[int, int, int]:
    """
    Applies a specified background pattern to an image.

    Args:
        img (Image.Image): The image to apply the pattern to.
        pattern_type (str): The type of pattern to apply.

    Returns:
        Tuple[int, int, int]: The background color used for the pattern.
    """
    draw: ImageDraw.Draw = ImageDraw.Draw(img)
    bg_color: Tuple[int, int, int] = get_random_color()
    
    if pattern_type == 'solid':
        draw.rectangle([0, 0, img.width, img.height], fill=bg_color)
    elif pattern_type == 'dots':
        for _ in range(20):
            x: int = random.randint(0, img.width)
            y: int = random.randint(0, img.height)
            draw.ellipse([x-2, y-2, x+2, y+2], fill=bg_color)
    elif pattern_type == 'grid':
        for x in range(0, img.width, 10):
            for y in range(0, img.height, 10):
                draw.rectangle([x, y, x+1, y+1], fill=bg_color)
    elif pattern_type == 'stripes':
        for y in range(0, img.height, 8):
            draw.line([(0, y), (img.width, y)], fill=bg_color)
    
    return bg_color

def generate_synthetic_gif(output_path: str, width: int=64, height: int=64, n_frames: int=10) -> Dict:
    """
    Generates a synthetic GIF with various animations.

    Args:
        output_path (str): The path to save the generated GIF.
        width (int): The width of the GIF.
        height (int): The height of the GIF.
        n_frames (int): The number of frames in the GIF.

    Returns:
        Dict: A dictionary containing the parameters used to generate the GIF.
    """
    frames: List[Image.Image] = []
    
    # Choose random animation parameters
    animation_type: str = random.choice([
        'moving_shape', 'growing_shape', 'bouncing_shape',
        'rotating_shape', 'color_changing', 'multiple_shapes'
    ])
    
    shape_type: str = random.choice(['circle', 'square', 'triangle', 'star'])
    primary_color: Tuple[int, int, int] = get_random_color()
    pattern_type: str = generate_random_pattern()
    
    # Animation specific parameters
    speed: float = random.uniform(0.5, 2.0)
    size_base: int = random.randint(10, 20)
    start_x: int = random.randint(0, width)
    start_y: int = random.randint(0, height)
    
    for i in range(n_frames):
        img: Image.Image = Image.new('RGB', (width, height), color='white')
        bg_color: Tuple[int, int, int] = apply_background_pattern(img, pattern_type)
        draw: ImageDraw.Draw = ImageDraw.Draw(img)
        
        if animation_type == 'moving_shape':
            x: int = (start_x + int(i * 5 * speed)) % width
            y: int = start_y
            if shape_type == 'circle':
                draw.ellipse([x-size_base, y-size_base, x+size_base, y+size_base], fill=primary_color)
            elif shape_type == 'square':
                draw.rectangle([x-size_base, y-size_base, x+size_base, y+size_base], fill=primary_color)
                
        elif animation_type == 'growing_shape':
            size: int = size_base + (i * 3)
            x: int = width//2 - size//2
            y: int = height//2 - size//2
            if shape_type == 'circle':
                draw.ellipse([x, y, x+size, y+size], fill=primary_color)
            elif shape_type == 'square':
                draw.rectangle([x, y, x+size, y+size], fill=primary_color)
                
        elif animation_type == 'bouncing_shape':
            x: int = width//2
            y: int = height//2 + int(20 * np.sin(i * speed * np.pi / 5))
            if shape_type == 'circle':
                draw.ellipse([x-size_base, y-size_base, x+size_base, y+size_base], fill=primary_color)
            elif shape_type == 'square':
                draw.rectangle([x-size_base, y-size_base, x+size_base, y+size_base], fill=primary_color)
                
        elif animation_type == 'rotating_shape':
            angle: float = i * 10 * speed
            img_rotate: Image.Image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw_rotate: ImageDraw.Draw = ImageDraw.Draw(img_rotate)
            if shape_type == 'square':
                draw_rotate.rectangle([-size_base, -size_base, size_base, size_base], fill=primary_color)
            img_rotate = img_rotate.rotate(angle)
            img.paste(img_rotate, (width//2, height//2), img_rotate)
            
        elif animation_type == 'color_changing':
            hue_shift: float = (i * 0.1 * speed) % 1.0
            current_color: Tuple[int, int, int] = tuple(int(x * 255) for x in colorsys.hsv_to_rgb(hue_shift, 1, 1))
            if shape_type == 'circle':
                draw.ellipse([width//2-size_base, height//2-size_base, 
                            width//2+size_base, height//2+size_base], fill=current_color)
            elif shape_type == 'square':
                draw.rectangle([width//2-size_base, height//2-size_base,
                              width//2+size_base, height//2+size_base], fill=current_color)
                
        elif animation_type == 'multiple_shapes':
            for j in range(3):
                x: int = (start_x + int((i + j*10) * speed)) % width
                y: int = start_y + j*15
                if shape_type == 'circle':
                    draw.ellipse([x-size_base//2, y-size_base//2, 
                                x+size_base//2, y+size_base//2], fill=primary_color)
                elif shape_type == 'square':
                    draw.rectangle([x-size_base//2, y-size_base//2,
                                  x+size_base//2, y+size_base//2], fill=primary_color)
        
        frames.append(img)
    
    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(100/speed),
        loop=0
    )
    
    return {
        'animation_type': animation_type,
        'shape_type': shape_type,
        'pattern_type': pattern_type,
        'speed': speed,
        'primary_color': primary_color,
        'bg_color': bg_color
    }

def get_color_name(rgb: Tuple[int, int, int]) -> str:
    """
    Converts an RGB tuple to the closest approximate color name.

    Args:
        rgb (Tuple[int, int, int]): An RGB color tuple.

    Returns:
        str: The closest color name.
    """
    colors: Dict[str, Tuple[int, int, int]] = {
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'purple': (128, 0, 128),
        'orange': (255, 165, 0),
        'pink': (255, 192, 203),
        'brown': (165, 42, 42),
        'gray': (128, 128, 128)
    }
    
    min_distance: float = float('inf')
    closest_color: str = 'unknown'
    
    for name, value in colors.items():
        distance: int = sum((a - b) ** 2 for a, b in zip(rgb, value))
        if distance < min_distance:
            min_distance = distance
            closest_color = name
    
    return closest_color

def generate_prompt(params: Dict) -> str:
    """
    Generates a detailed and varied text prompt based on animation parameters.

    Args:
        params (Dict): A dictionary of parameters for the generated animation.

    Returns:
        str: The generated text prompt.
    """
    action_templates: Dict[str, List[str]] = {
        'moving_shape': [
            "A {color} {shape} sliding smoothly across a {pattern} {bg_color} background",
            "An animated {color} {shape} traveling from left to right over {pattern} {bg_color} backdrop",
            "{color} {shape} in motion against a {pattern} {bg_color} scene"
        ],
        'growing_shape': [
            "A {color} {shape} expanding gradually on a {pattern} {bg_color} canvas",
            "An enlarging {color} {shape} centered on a {pattern} {bg_color} background",
            "Dynamic animation of a {color} {shape} growing in size with {pattern} {bg_color} backdrop"
        ],
        'bouncing_shape': [
            "A {color} {shape} bouncing rhythmically on a {pattern} {bg_color} surface",
            "Animated {color} {shape} moving up and down against {pattern} {bg_color} background",
            "Playful {color} {shape} bouncing with {pattern} {bg_color} backdrop"
        ],
        'rotating_shape': [
            "A {color} {shape} spinning smoothly on a {pattern} {bg_color} background",
            "Rotating {color} {shape} animation with {pattern} {bg_color} backdrop",
            "Dynamic {color} {shape} turning in space against {pattern} {bg_color} scene"
        ],
        'color_changing': [
            "A mesmerizing {shape} shifting through rainbow colors on {pattern} {bg_color} background",
            "Color-morphing {shape} display against {pattern} {bg_color} backdrop",
            "Prismatic {shape} animation with changing hues on {pattern} {bg_color} canvas"
        ],
        'multiple_shapes': [
            "Three {color} {shape}s moving in parallel on {pattern} {bg_color} background",
            "Trio of animated {color} {shape}s with {pattern} {bg_color} backdrop",
            "Synchronized {color} {shape}s dancing across {pattern} {bg_color} scene"
        ]
    }
    
    speed_desc: str = "quickly" if params['speed'] > 1.5 else "slowly" if params['speed'] < 0.8 else "steadily"
    
    template: str = random.choice(action_templates[params['animation_type']])
    prompt: str = template.format(
        color=get_color_name(params['primary_color']),
        shape=params['shape_type'],
        pattern=params['pattern_type'],
        bg_color=get_color_name(params['bg_color'])
    )
    
    # Add speed description if relevant
    if params['animation_type'] in ['moving_shape', 'rotating_shape', 'bouncing_shape']:
        prompt = prompt.replace('moving', f'moving {speed_desc}')
        prompt = prompt.replace('spinning', f'spinning {speed_desc}')
        prompt = prompt.replace('bouncing', f'bouncing {speed_desc}')
    
    return prompt

def generate_dataset(n_samples: int=10000) -> str:
    """
    Generates a complete synthetic video dataset, including GIFs and corresponding prompts.

    Args:
        n_samples (int): The number of samples to generate.

    Returns:
        str: The base directory path where the dataset is saved.
    """
    training_dir: str = create_directory_structure()
    
    print(f"Generating {n_samples} samples...")
    
    # Keep track of generated prompts to ensure uniqueness
    generated_prompts: set = set()
    
    for i in tqdm(range(n_samples), desc="Generating samples"):
        # Generate unique identifier
        sample_id: str = f"{i:05d}"
        
        # Keep generating until we get a unique prompt
        while True:
            # Generate and save GIF
            gif_path: str = os.path.join(training_dir, f"{sample_id}.gif")
            params: Dict = generate_synthetic_gif(gif_path)
            
            # Generate prompt
            prompt: str = generate_prompt(params)
            
            # Check if this exact prompt has been generated before
            if prompt not in generated_prompts:
                generated_prompts.add(prompt)
                break
        
        # Save prompt
        prompt_path: str = os.path.join(training_dir, f"{sample_id}.txt")
        with open(prompt_path, 'w') as f:
            f.write(prompt)
    
    print("Dataset generation complete!")
    return training_dir

# Generate the dataset
if __name__ == "__main__":
    output_dir: str = generate_dataset(n_samples=10000)
    print(f"Dataset saved to: {output_dir}")