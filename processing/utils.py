"""
Utility functions for image processing operations.
This module contains shared helper functions used across different processing modules.
"""

import numpy as np
from PIL import Image
from typing import Union, Tuple


def _ensure_numpy():
    if np is None:
        raise ImportError("NumPy is required for image processing operations. Please install 'numpy'.")


def _pil_to_numpy(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array (RGB)"""
    return np.array(pil_img)


def _numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image"""
    if arr.ndim == 2:
        return Image.fromarray(arr, mode='L')
    elif arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr, mode='RGB')
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")


def _apply_color_tint(img: np.ndarray, r_factor: float = 1.0, g_factor: float = 1.0,
                     b_factor: float = 1.0, bias: int = 0) -> np.ndarray:
    """Apply color tint using optimized NumPy operations"""
    _ensure_numpy()

    # Ensure image is float32 for calculations
    img_f = img.astype(np.float32)

    factors = np.array([r_factor, g_factor, b_factor])
    tinted = img_f * factors

    # Apply bias if specified
    if bias != 0:
        tinted += bias

    # Clip to valid range and convert back to uint8
    return np.clip(tinted, 0, 255).astype(np.uint8)