"""
Arithmetic operations for image processing.
This module contains functions for basic arithmetic operations on images.
"""

import numpy as np
from PyQt5.QtWidgets import QInputDialog, QFileDialog, QMessageBox
from .utils import _ensure_numpy


def add_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Add two images pixel-wise"""
    _ensure_numpy()
    # Ensure images have the same dimensions
    img1, img2 = ensure_same_dimensions(img1, img2)
    result = img1.astype(np.float32) + img2.astype(np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


def add_constant(img: np.ndarray, constant: float) -> np.ndarray:
    """Add a constant value to image"""
    _ensure_numpy()
    result = img.astype(np.float32) + constant
    return np.clip(result, 0, 255).astype(np.uint8)


def subtract_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Subtract img2 from img1 pixel-wise"""
    _ensure_numpy()
    # Ensure images have the same dimensions
    img1, img2 = ensure_same_dimensions(img1, img2)
    result = img1.astype(np.float32) - img2.astype(np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


def subtract_constant(img: np.ndarray, constant: float) -> np.ndarray:
    """Subtract a constant value from image"""
    _ensure_numpy()
    result = img.astype(np.float32) - constant
    return np.clip(result, 0, 255).astype(np.uint8)


def absolute_difference(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Compute absolute difference between two images"""
    _ensure_numpy()
    # Ensure images have the same dimensions
    img1, img2 = ensure_same_dimensions(img1, img2)
    result = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    return np.clip(result, 0, 255).astype(np.uint8)


def multiply_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Multiply two images pixel-wise"""
    _ensure_numpy()
    # Ensure images have the same dimensions
    img1, img2 = ensure_same_dimensions(img1, img2)
    result = img1.astype(np.float32) * img2.astype(np.float32) / 255.0
    return np.clip(result, 0, 255).astype(np.uint8)


def multiply_constant(img: np.ndarray, constant: float) -> np.ndarray:
    """Multiply image by a constant value"""
    _ensure_numpy()
    result = img.astype(np.float32) * constant
    return np.clip(result, 0, 255).astype(np.uint8)


def divide_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Divide img1 by img2 pixel-wise"""
    _ensure_numpy()
    # Ensure images have the same dimensions
    img1, img2 = ensure_same_dimensions(img1, img2)
    # Avoid division by zero
    img2_float = img2.astype(np.float32)
    img2_float[img2_float == 0] = 1  # Replace zeros with 1 to avoid division by zero
    result = img1.astype(np.float32) / img2_float * 255.0
    return np.clip(result, 0, 255).astype(np.uint8)


def divide_constant(img: np.ndarray, constant: float) -> np.ndarray:
    """Divide image by a constant value"""
    _ensure_numpy()
    if constant == 0:
        raise ValueError("Cannot divide by zero")
    result = img.astype(np.float32) / constant
    return np.clip(result, 0, 255).astype(np.uint8)


def blend_images(img1: np.ndarray, img2: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Blend two images using linear combination: alpha*img1 + beta*img2"""
    _ensure_numpy()
    # Ensure images have the same dimensions
    img1, img2 = ensure_same_dimensions(img1, img2)
    result = alpha * img1.astype(np.float32) + beta * img2.astype(np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


def get_second_image() -> np.ndarray:
    """Helper function to get second image from file dialog"""
    from processing.qt import pixmap_to_numpy
    from PyQt5.QtGui import QPixmap

    file_path, _ = QFileDialog.getOpenFileName(
        None,
        'Select Second Image',
        '',
        'Images (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff);;All Files (*)'
    )
    if not file_path:
        return None

    pixmap = QPixmap(file_path)
    if pixmap.isNull():
        QMessageBox.warning(None, 'Error', 'Cannot load the selected image.')
        return None

    return pixmap_to_numpy(pixmap)


def get_constant_value(prompt: str, default: float = 0.0) -> float:
    """Helper function to get constant value from user input"""
    value, ok = QInputDialog.getDouble(None, 'Enter Value', prompt, default, -255.0, 255.0, 2)
    if not ok:
        return None
    return value


def get_blend_parameters() -> tuple:
    """Helper function to get alpha and beta values for blending"""
    alpha, ok1 = QInputDialog.getDouble(None, 'Blend Parameters', 'Alpha (weight for first image):', 0.5, 0.0, 1.0, 2)
    if not ok1:
        return None, None

    beta, ok2 = QInputDialog.getDouble(None, 'Blend Parameters', 'Beta (weight for second image):', 0.5, 0.0, 1.0, 2)
    if not ok2:
        return None, None

    return alpha, beta


def resize_image_to_match(img: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resize image to match target dimensions using OpenCV"""
    _ensure_numpy()
    import cv2
    target_height, target_width = target_shape[:2]
    resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return resized


def ensure_same_dimensions(img1: np.ndarray, img2: np.ndarray) -> tuple:
    """Ensure both images have the same dimensions by resizing the second to match the first"""
    if img1.shape == img2.shape:
        return img1, img2

    # Resize img2 to match img1 dimensions
    resized_img2 = resize_image_to_match(img2, img1.shape)
    return img1, resized_img2