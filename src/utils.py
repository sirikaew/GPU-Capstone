from PIL import Image
import numpy as np

def load_image(path):
    """Load image and return float32 array in [0,1]."""
    img = Image.open(path).convert('RGB')
    return np.asarray(img, dtype=np.float32) / 255.0

def save_image(arr, path):
    """Save float array (0‑1) as image."""
    arr = (np.clip(arr, 0, 1) * 255).astype('uint8')
    Image.fromarray(arr).save(path)
