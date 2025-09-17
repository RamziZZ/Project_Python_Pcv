"""
Image filtering operations for image processing.
This module contains all filter-related transformations including convolution-based filters.
"""

import numpy as np
import cv2
from .utils import _ensure_numpy


def identity(img: np.ndarray) -> np.ndarray:
    """Identity filter - returns the image unchanged"""
    _ensure_numpy()
    return img.copy()


def _to_grayscale_if_needed(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale if image is RGB"""
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def _to_rgb_stack(gray_img: np.ndarray) -> np.ndarray:
    """Convert grayscale to RGB stack for consistency"""
    return np.stack([gray_img, gray_img, gray_img], axis=2)


def edge_detection_1(img: np.ndarray) -> np.ndarray:
    """Edge detection using Sobel-like kernel"""
    _ensure_numpy()
    gray = _to_grayscale_if_needed(img)

    # Simple edge detection kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]], dtype=np.float32)

    edges = cv2.filter2D(gray, -1, kernel)
    return _to_rgb_stack(edges)


def edge_detection_2(img: np.ndarray) -> np.ndarray:
    """Edge detection using Prewitt kernel"""
    _ensure_numpy()
    gray = _to_grayscale_if_needed(img)

    # Prewitt horizontal kernel
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]], dtype=np.float32)

    # Prewitt vertical kernel
    kernel_y = np.array([[-1, -1, -1],
                         [0,  0,  0],
                         [1,  1,  1]], dtype=np.float32)

    edges_x = cv2.filter2D(gray, -1, kernel_x)
    edges_y = cv2.filter2D(gray, -1, kernel_y)
    edges = cv2.addWeighted(edges_x, 0.5, edges_y, 0.5, 0)

    return _to_rgb_stack(edges)


def edge_detection_3(img: np.ndarray) -> np.ndarray:
    """Edge detection using Laplacian kernel"""
    _ensure_numpy()
    gray = _to_grayscale_if_needed(img)

    # Laplacian kernel
    kernel = np.array([[0,  1, 0],
                       [1, -4, 1],
                       [0,  1, 0]], dtype=np.float32)

    edges = cv2.filter2D(gray, -1, kernel)
    # Convert to absolute values and scale
    edges = np.abs(edges)
    edges = np.clip(edges * 2, 0, 255).astype(np.uint8)

    return _to_rgb_stack(edges)


def sharpen(img: np.ndarray) -> np.ndarray:
    """Sharpen filter using unsharp masking technique"""
    _ensure_numpy()
    if img.ndim == 3 and img.shape[2] == 3:
        # Apply sharpening to each channel
        channels = cv2.split(img)
        sharpened_channels = []
        for ch in channels:
            sharpened = _sharpen_channel(ch)
            sharpened_channels.append(sharpened)
        return cv2.merge(sharpened_channels)
    else:
        return _sharpen_channel(img)


def _sharpen_channel(channel: np.ndarray) -> np.ndarray:
    """Sharpen a single channel"""
    # Sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]], dtype=np.float32)

    sharpened = cv2.filter2D(channel, -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def gaussian_blur_3x3(img: np.ndarray) -> np.ndarray:
    """Gaussian blur with 3x3 kernel"""
    _ensure_numpy()
    return cv2.GaussianBlur(img, (3, 3), 0)


def gaussian_blur_5x5(img: np.ndarray) -> np.ndarray:
    """Gaussian blur with 5x5 kernel"""
    _ensure_numpy()
    return cv2.GaussianBlur(img, (5, 5), 0)


def unsharp_masking(img: np.ndarray) -> np.ndarray:
    """Unsharp masking for image sharpening"""
    _ensure_numpy()
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Calculate mask
    mask = cv2.subtract(img, blurred)

    # Add mask back to original with weight
    sharpened = cv2.addWeighted(img, 1.5, mask, 0.5, 0)
    return sharpened


def average_filter(img: np.ndarray) -> np.ndarray:
    """Average/mean filter"""
    _ensure_numpy()
    # 3x3 average kernel
    kernel = np.ones((3, 3), np.float32) / 9
    return cv2.filter2D(img, -1, kernel)


def low_pass_filter(img: np.ndarray) -> np.ndarray:
    """Low pass filter - removes high frequency components"""
    _ensure_numpy()
    # Simple low pass filter using Gaussian blur
    return cv2.GaussianBlur(img, (5, 5), 1.0)


def high_pass_filter(img: np.ndarray) -> np.ndarray:
    """High pass filter - removes low frequency components"""
    _ensure_numpy()
    # Apply low pass filter
    low_pass = cv2.GaussianBlur(img, (5, 5), 1.0)

    # Subtract low pass from original
    high_pass = cv2.subtract(img, low_pass)

    # Add offset to make result visible
    high_pass = cv2.add(high_pass, 128)

    return high_pass


def bandstop_filter(img: np.ndarray) -> np.ndarray:
    """Bandstop filter - removes mid-range frequencies"""
    _ensure_numpy()
    # Apply two Gaussian blurs with different sigma
    blur_small = cv2.GaussianBlur(img, (3, 3), 0.5)
    blur_large = cv2.GaussianBlur(img, (9, 9), 2.0)

    # Subtract the difference
    bandstop = cv2.subtract(blur_small, blur_large)

    # Add offset
    bandstop = cv2.add(bandstop, 128)

    return bandstop


def prewitt(img: np.ndarray) -> np.ndarray:
    """Prewitt edge detection operator"""
    _ensure_numpy()
    gray = _to_grayscale_if_needed(img)

    # Prewitt horizontal kernel
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]], dtype=np.float32)

    # Prewitt vertical kernel
    kernel_y = np.array([[-1, -1, -1],
                         [0,  0,  0],
                         [1,  1,  1]], dtype=np.float32)

    edges_x = cv2.filter2D(gray, -1, kernel_x)
    edges_y = cv2.filter2D(gray, -1, kernel_y)
    edges = cv2.addWeighted(edges_x, 0.5, edges_y, 0.5, 0)

    return _to_rgb_stack(edges)


def sobel(img: np.ndarray) -> np.ndarray:
    """Sobel edge detection operator"""
    _ensure_numpy()
    gray = _to_grayscale_if_needed(img)

    # Sobel horizontal kernel
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=np.float32)

    # Sobel vertical kernel
    kernel_y = np.array([[-1, -2, -1],
                         [0,  0,  0],
                         [1,  2,  1]], dtype=np.float32)

    edges_x = cv2.filter2D(gray, -1, kernel_x)
    edges_y = cv2.filter2D(gray, -1, kernel_y)
    edges = cv2.addWeighted(edges_x, 0.5, edges_y, 0.5, 0)

    return _to_rgb_stack(edges)