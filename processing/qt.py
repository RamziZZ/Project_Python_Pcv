import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QMessageBox
from PyQt5.QtGui import QImage, QPixmap


def pixmap_to_numpy(pix: QPixmap) -> np.ndarray:
    """Convert QPixmap to numpy array in RGB format"""
    img = pix.toImage()
    if img.isNull():
        raise ValueError("QImage is null")

    # Ensure ARGB32 backing for predictable byte layout (BGRA in memory)
    img = img.convertToFormat(img.Format_ARGB32)

    w = img.width()
    h = img.height()
    ptr = img.bits()
    ptr.setsize(img.byteCount())
    arr = np.frombuffer(ptr, np.uint8).reshape((h, img.bytesPerLine() // 4, 4))
    # BGRA -> RGB
    rgb = arr[:, :w, 0:3][:, :, ::-1]
    return rgb.copy()


def numpy_to_pixmap(arr: np.ndarray) -> QPixmap:
    """Convert numpy array to QPixmap"""
    if arr.ndim == 2:
        h, w = arr.shape
        # Convert to 3-channel grayscale for display
        arr_rgb = np.stack([arr, arr, arr], axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        h, w, _ = arr.shape
        arr_rgb = arr
    else:
        raise ValueError("Unsupported array shape for image: %r" % (arr.shape,))

    # Ensure contiguous in memory
    arr_rgb = np.ascontiguousarray(arr_rgb)
    # Create QImage from numpy array
    qimg = QImage(arr_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    # Deep copy to detach from numpy memory
    return QPixmap.fromImage(qimg.copy())


def show_histogram(image: np.ndarray, title: str):
    """Display histogram of the image in a dialog"""
    fig, ax = plt.subplots()
    if image.ndim == 3 and image.shape[2] == 3:
        # RGB image - show separate histograms for each channel
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            ax.plot(hist, color=color, label=color)
        ax.legend()
    else:
        # Grayscale or single channel
        if image.ndim == 3:
            # Convert to grayscale for histogram
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        ax.plot(hist, color='black')

    ax.set_title(title)
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')

    # Create dialog
    dialog = QDialog()
    dialog.setWindowTitle(title)
    layout = QVBoxLayout()
    canvas = FigureCanvas(fig)
    layout.addWidget(canvas)
    dialog.setLayout(layout)
    dialog.exec_()


def show_input_histogram(input_pixmap: QPixmap):
    """Show histogram for input image"""
    arr = pixmap_to_numpy(input_pixmap)
    show_histogram(arr, "Input Histogram")


def show_output_histogram(output_pixmap: QPixmap):
    """Show histogram for output image"""
    if output_pixmap is None or output_pixmap.isNull():
        QMessageBox.information(None, 'Info', 'No output image available.')
        return
    arr = pixmap_to_numpy(output_pixmap)
    show_histogram(arr, "Output Histogram")


def show_input_output_histogram(input_pixmap: QPixmap, output_pixmap: QPixmap):
    """Show histograms for both input and output images side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Input histogram
    arr_in = pixmap_to_numpy(input_pixmap)
    if arr_in.ndim == 3 and arr_in.shape[2] == 3:
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            hist = cv2.calcHist([arr_in], [i], None, [256], [0, 256])
            ax1.plot(hist, color=color, label=color)
        ax1.legend()
    else:
        if arr_in.ndim == 3:
            gray = cv2.cvtColor(arr_in, cv2.COLOR_RGB2GRAY)
        else:
            gray = arr_in
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        ax1.plot(hist, color='black')
    ax1.set_title('Input Histogram')
    ax1.set_xlabel('Pixel Value')
    ax1.set_ylabel('Frequency')

    # Output histogram
    if output_pixmap is not None and not output_pixmap.isNull():
        arr_out = pixmap_to_numpy(output_pixmap)
        if arr_out.ndim == 3 and arr_out.shape[2] == 3:
            colors = ['red', 'green', 'blue']
            for i, color in enumerate(colors):
                hist = cv2.calcHist([arr_out], [i], None, [256], [0, 256])
                ax2.plot(hist, color=color, label=color)
            ax2.legend()
        else:
            if arr_out.ndim == 3:
                gray = cv2.cvtColor(arr_out, cv2.COLOR_RGB2GRAY)
            else:
                gray = arr_out
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            ax2.plot(hist, color='black')
        ax2.set_title('Output Histogram')
        ax2.set_xlabel('Pixel Value')
        ax2.set_ylabel('Frequency')
    else:
        ax2.text(0.5, 0.5, 'No Output Available', transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title('Output Histogram')

    # Create dialog
    dialog = QDialog()
    dialog.setWindowTitle("Input vs Output Histogram")
    layout = QVBoxLayout()
    canvas = FigureCanvas(fig)
    layout.addWidget(canvas)
    dialog.setLayout(layout)
    dialog.exec_()
