"""
Color manipulation operations for image processing.
This module contains all color-related transformations.
"""

import numpy as np
from typing import Tuple
from .utils import _ensure_numpy, _apply_color_tint


def rgb_yellow(img: np.ndarray) -> np.ndarray:
    """Apply yellow tint (boost red and green channels)"""
    return _apply_color_tint(img, r_factor=1.2, g_factor=1.2, b_factor=0.9)


def rgb_cyan(img: np.ndarray) -> np.ndarray:
    """Apply cyan tint (boost green and blue channels)"""
    return _apply_color_tint(img, r_factor=0.9, g_factor=1.2, b_factor=1.2)


def rgb_orange(img: np.ndarray) -> np.ndarray:
    """Apply orange tint (strong red, slight green, reduce blue)"""
    return _apply_color_tint(img, r_factor=1.25, g_factor=1.05, b_factor=0.85)


def rgb_purple(img: np.ndarray) -> np.ndarray:
    """Apply purple tint (boost red and blue, reduce green)"""
    return _apply_color_tint(img, r_factor=1.2, g_factor=0.85, b_factor=1.2)


def rgb_grey(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale using luminance method"""
    return to_grayscale_luminance(img)


def rgb_brown(img: np.ndarray) -> np.ndarray:
    """Apply brown tint (warm colors with darker bias)"""
    return _apply_color_tint(img, r_factor=1.2, g_factor=1.0, b_factor=0.85, bias=-15)


def rgb_red(img: np.ndarray) -> np.ndarray:
    """Apply red tint (emphasize red channel)"""
    return _apply_color_tint(img, r_factor=1.25, g_factor=0.9, b_factor=0.9)


# Grayscale conversion methods
def to_grayscale_average(img: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale using average method"""
    _ensure_numpy()
    if img.ndim == 2:
        return img

    # Calculate average of RGB channels (R+G+B)/3
    img_f = img.astype(np.float32)
    gray = np.mean(img_f, axis=2)
    gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)

    # Return as RGB for consistency
    return np.stack([gray_uint8, gray_uint8, gray_uint8], axis=2)


def to_grayscale_lightness(img: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale using lightness method (average of min/max)"""
    _ensure_numpy()
    if img.ndim == 2:
        return img

    # Lightness method: average of min and max RGB values per pixel
    img_f = img.astype(np.float32)
    max_rgb = np.max(img_f, axis=2)
    min_rgb = np.min(img_f, axis=2)
    gray = (max_rgb + min_rgb) / 2.0
    gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)

    # Return as RGB for consistency
    return np.stack([gray_uint8, gray_uint8, gray_uint8], axis=2)


def to_grayscale_luminance(img: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale using optimized NumPy luminance method"""
    _ensure_numpy()
    if img.ndim == 2:
        return img

    img_f = img.astype(np.float32)
    gray = np.dot(img_f, [0.299, 0.587, 0.114])
    gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)

    return np.stack([gray_uint8, gray_uint8, gray_uint8], axis=2)