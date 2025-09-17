"""
Image enhancement operations for brightness, contrast, and other adjustments.
This module contains all enhancement-related transformations.
"""

import numpy as np
from typing import Tuple
from .utils import _ensure_numpy


def invert(img: np.ndarray) -> np.ndarray:
    """Invert image colors using optimized NumPy operations"""
    _ensure_numpy()
    return 255 - img


def log_brightness(img: np.ndarray) -> np.ndarray:
    """Apply logarithmic brightness enhancement"""
    _ensure_numpy()
    img_f = img.astype(np.float32)
    # Normalize to 0..1
    img_n = img_f / 255.0
    log_img = np.log1p(img_n)
    log_img /= log_img.max() if log_img.max() > 0 else 1.0
    out = (log_img * 255.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def gamma_correction(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply gamma correction using optimized NumPy operations"""
    _ensure_numpy()
    gamma = max(1e-6, float(gamma))

    # Early return for no-op case
    if abs(gamma - 1.0) < 1e-6:
        return img.copy()

    # Use lookup table for better performance on repeated calls
    inv_gamma = 1.0 / gamma
    lut = np.power(np.arange(256) / 255.0, inv_gamma) * 255.0
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    # Apply using lookup table (faster than power function per pixel)
    return lut[img]


def brightness_contrast(img: np.ndarray, brightness: float = 0.0, contrast: float = 1.0) -> np.ndarray:
    """Adjust brightness and contrast using optimized NumPy operations"""
    _ensure_numpy()

    # Convert to float32 for calculations
    img_f = img.astype(np.float32)

    # Apply brightness adjustment
    if brightness != 0:
        img_f += brightness

    # Apply contrast adjustment using the formula: (img - 128) * contrast + 128
    if contrast != 1.0:
        img_f = (img_f - 128.0) * contrast + 128.0

    # Clip to valid range and convert back to uint8
    return np.clip(img_f, 0, 255).astype(np.uint8)

