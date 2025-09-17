import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QAction, QMenu
from PyQt5 import uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QGraphicsScene, QInputDialog
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtCore import Qt, QRectF

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(BASE_DIR, 'ui')

class TentangWindow(QWidget):
    def __init__(self, parent=None, on_close=None):
        super().__init__(parent)
        uic.loadUi(os.path.join(UI_DIR, 'TentangWindow.ui'), self)
        self._on_close = on_close

    def closeEvent(self, event):
        if self._on_close:
            self._on_close()
        event.accept()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(os.path.join(UI_DIR, 'MainWindow.ui'), self)

        # Setup scenes for graphics views (left/right). Left is used for input display.
        self._left_scene = QGraphicsScene(self.graphicsView)
        self.graphicsView.setScene(self._left_scene)
        self.graphicsView.setRenderHints(self.graphicsView.renderHints() |
                                         QPainter.Antialiasing |
                                         QPainter.SmoothPixmapTransform)
        # Right scene for processed output
        self._right_scene = QGraphicsScene(self.graphicsView_2)
        self.graphicsView_2.setScene(self._right_scene)
        self.graphicsView_2.setRenderHints(self.graphicsView_2.renderHints() |
                                           QPainter.Antialiasing |
                                           QPainter.SmoothPixmapTransform)

        # Internal state
        self._input_pixmap: QPixmap = None
        self._output_pixmap: QPixmap = None

        buka_action = self.findChild(QAction, 'actionBuka')
        if buka_action is not None:
            buka_action.triggered.connect(self.open_image)

        simpan_sebagai = self.findChild(QAction, 'actionSimpan_Sebagai')
        if simpan_sebagai is not None:
            simpan_sebagai.triggered.connect(self.save_output_as)

        keluar_action = self.findChild(QAction, 'actionKeluar')
        if keluar_action is not None:
            keluar_action.triggered.connect(self.close)

        # Tentang
        tentang_action = self.findChild(QAction, 'actionTentang')
        if tentang_action is not None:
            tentang_action.triggered.connect(self.show_tentang)
        else:
            tentang_menu = self.findChild(QMenu, 'menuTentang')
            if tentang_menu is not None:
                tentang_menu.aboutToShow.connect(self.show_tentang)
            else:
                print("QAction 'Tentang' tidak ditemukan di menu bar.")
        self.tentang_window = None
        self._current_image_path = None

        # Wire Colors menu actions
        self._wire_colors_actions()

        # Wire Image Processing menu actions
        self._wire_image_processing_actions()

        # Wire View menu actions
        self._wire_view_actions()

        # Wire Arithmetical Operation menu actions
        self._wire_arithmetic_actions()

        # Wire Filter menu actions
        self._wire_filter_actions()

        # Wire standalone Edge Detection menu actions
        self._wire_edge_detection_actions()

    def _display_pixmap_on_left(self, pixmap: QPixmap):
        if pixmap.isNull():
            QMessageBox.warning(self, 'Gagal Membuka', 'Gambar tidak valid atau tidak dapat dimuat.')
            return
        self._left_scene.clear()
        item = self._left_scene.addPixmap(pixmap)
        self._left_scene.setSceneRect(QRectF(pixmap.rect()))
        # Fit to view while keeping aspect ratio
        self.graphicsView.fitInView(item, Qt.KeepAspectRatio)

    def _display_pixmap_on_right(self, pixmap: QPixmap):
        if pixmap is None or pixmap.isNull():
            return
        self._right_scene.clear()
        item = self._right_scene.addPixmap(pixmap)
        self._right_scene.setSceneRect(QRectF(pixmap.rect()))
        self.graphicsView_2.fitInView(item, Qt.KeepAspectRatio)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            'Buka Gambar',
            '',
            'Images (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff);;All Files (*)'
        )
        if not file_path:
            return
        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            QMessageBox.warning(self, 'Gagal Membuka', 'Tidak dapat memuat file gambar yang dipilih.')
            return
        self._current_image_path = file_path
        self._input_pixmap = pixmap
        self._output_pixmap = None
        self._display_pixmap_on_left(pixmap)
        self._right_scene.clear()
        self.statusBar().showMessage(f'Terbuka: {os.path.basename(file_path)}', 5000)

    def save_output_as(self):
        if self._output_pixmap is None or self._output_pixmap.isNull():
            QMessageBox.information(self, 'Simpan', 'Tidak ada hasil untuk disimpan. Lakukan pemrosesan terlebih dahulu.')
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            'Simpan Hasil Sebagai',
            '',
            'PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF (*.tif *.tiff)'
        )
        if not file_path:
            return

        # Ensure file has proper extension based on selected filter
        if selected_filter.startswith('PNG'):
            if not file_path.lower().endswith('.png'):
                file_path += '.png'
        elif selected_filter.startswith('JPEG'):
            if not any(file_path.lower().endswith(ext) for ext in ['.jpg', '.jpeg']):
                file_path += '.jpg'
        elif selected_filter.startswith('BMP'):
            if not file_path.lower().endswith('.bmp'):
                file_path += '.bmp'
        elif selected_filter.startswith('TIFF'):
            if not any(file_path.lower().endswith(ext) for ext in ['.tif', '.tiff']):
                file_path += '.tif'

        # Ensure directory exists
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except Exception as e:
                QMessageBox.warning(self, 'Simpan', f'Gagal membuat direktori: {e}')
                return

        # Try to save with better error handling
        try:
            # Ensure pixmap is valid
            if self._output_pixmap.isNull():
                QMessageBox.warning(self, 'Simpan', 'Gambar output tidak valid.')
                return

            # Try to save
            success = self._output_pixmap.save(file_path)

            if success:
                self.statusBar().showMessage(f'Tersimpan: {os.path.basename(file_path)}', 5000)
            else:
                # Try alternative approach for JPEG
                if file_path.lower().endswith(('.jpg', '.jpeg')):
                    # For JPEG, try converting to RGB first if needed
                    temp_pixmap = self._output_pixmap
                    success = temp_pixmap.save(file_path, 'JPEG', 95)  # 95% quality

                if not success:
                    QMessageBox.warning(self, 'Simpan', f'Gagal menyimpan gambar ke: {file_path}\nPeriksa izin file dan direktori.')

        except Exception as e:
            QMessageBox.warning(self, 'Simpan', f'Error saat menyimpan: {str(e)}')

    def _reset_tentang_window(self):
        self.tentang_window = None

    # ---- Colors actions handling ----
    def _require_input(self) -> bool:
        if self._input_pixmap is None or self._input_pixmap.isNull():
            QMessageBox.information(self, 'Info', 'Buka gambar terlebih dahulu (File -> Buka).')
            return False
        return True

    def _apply_and_show(self, fn, checked=False, *args, **kwargs):
        """Run fn on input image (numpy), show result on right."""
        if not self._require_input():
            return
        from processing.qt import pixmap_to_numpy, numpy_to_pixmap
        import numpy as np
        arr = pixmap_to_numpy(self._input_pixmap)
        try:
            out = fn(arr, *args, **kwargs)
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Gagal memproses gambar: {e}')
            return
        # Ensure uint8 RGB
        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)
        if out.ndim == 2:
            out = np.stack([out, out, out], axis=2)
        out_pix = numpy_to_pixmap(out)
        # Ensure pixmap is valid before storing
        if out_pix.isNull():
            QMessageBox.warning(self, 'Error', 'Gagal membuat gambar output yang valid.')
            return
        self._output_pixmap = out_pix
        self._display_pixmap_on_right(out_pix)

    def _wire_colors_actions(self):
        from functools import partial
        from processing import ops

        # RGB tints
        mapping = {
            'actionYellow': ops.rgb_yellow,
            'actionOrange': ops.rgb_orange,
            'actionCyan': ops.rgb_cyan,
            'actionPurple': ops.rgb_purple,
            'actionGrey': ops.rgb_grey,
            'actionBrown': ops.rgb_brown,
            'actionRed': ops.rgb_red,
        }
        for name, fn in mapping.items():
            act = self.findChild(QAction, name)
            if act:
                act.triggered.connect(partial(self._apply_and_show, fn))

        # RGB -> Grayscale
        gray_map = {
            'actionAverage_2': ops.to_grayscale_average,
            'actionLightness_2': ops.to_grayscale_lightness,
            'actionLuminance_2': ops.to_grayscale_luminance,
        }
        for name, fn in gray_map.items():
            act = self.findChild(QAction, name)
            if act:
                act.triggered.connect(partial(self._apply_and_show, fn))

        # Invers
        act_inv = self.findChild(QAction, 'actionInvers')
        if act_inv:
            act_inv.triggered.connect(partial(self._apply_and_show, ops.invert))

        # Log Brightness
        act_log = self.findChild(QAction, 'actionLog_Brightness')
        if act_log:
            act_log.triggered.connect(partial(self._apply_and_show, ops.log_brightness))

        # Gamma Correction (ask for gamma)
        act_gamma = self.findChild(QAction, 'actionGamma_Correction')
        if act_gamma:
            act_gamma.triggered.connect(self._on_gamma)

        # Brightness -> Contrast (only contrast factor)
        act_contrast = self.findChild(QAction, 'actionContrast')
        if act_contrast:
            act_contrast.triggered.connect(self._on_contrast_only)

        # Brightness - Contrast (both)
        act_bc = self.findChild(QAction, 'actionBrightness_Contrast')
        if act_bc:
            act_bc.triggered.connect(self._on_brightness_contrast)

        # Bit Depth 1..7
        for bits in range(1, 8):
            act = self.findChild(QAction, f'action{bits}_bit')
            if act:
                act.triggered.connect(partial(self._apply_and_show, ops.bit_depth, bits=bits))

    # Parameterized handlers
    def _on_gamma(self):
        if not self._require_input():
            return
        gamma, ok = QInputDialog.getDouble(self, 'Gamma Correction', 'Gamma (e.g., 0.5..3.0):', 1.0, 0.01, 10.0, 2)
        if not ok:
            return
        from processing import ops
        self._apply_and_show(ops.gamma_correction, gamma=gamma)

    def _on_contrast_only(self):
        if not self._require_input():
            return
        contrast, ok = QInputDialog.getDouble(self, 'Contrast', 'Factor (>0, e.g., 1.2):', 1.2, 0.01, 10.0, 2)
        if not ok:
            return
        from processing import ops
        self._apply_and_show(ops.brightness_contrast, brightness=0.0, contrast=contrast)

    def _on_brightness_contrast(self):
        if not self._require_input():
            return
        brightness, ok1 = QInputDialog.getInt(self, 'Brightness - Contrast', 'Brightness (-255..255):', 0, -255, 255, 1)
        if not ok1:
            return
        contrast, ok2 = QInputDialog.getDouble(self, 'Brightness - Contrast', 'Contrast (>0):', 1.0, 0.01, 10.0, 2)
        if not ok2:
            return
        from processing import ops
        self._apply_and_show(ops.brightness_contrast, brightness=brightness, contrast=contrast)

    def _wire_image_processing_actions(self):
        from functools import partial
        from processing import ops

        # Histogram Equalization
        act_he = self.findChild(QAction, 'actionHistogram_Equalization')
        if act_he:
            act_he.triggered.connect(partial(self._apply_and_show, ops.histogram_equalization))

        # Fuzzy HE RGB
        act_fuzzy_rgb = self.findChild(QAction, 'actionFuzzy_HE_RGB')
        if act_fuzzy_rgb:
            act_fuzzy_rgb.triggered.connect(partial(self._apply_and_show, ops.fuzzy_histogram_equalization_rgb))

        # Fuzzy Grayscale
        act_fuzzy_gray = self.findChild(QAction, 'actionFuzzy_Grayscale')
        if act_fuzzy_gray:
            act_fuzzy_gray.triggered.connect(partial(self._apply_and_show, ops.fuzzy_histogram_equalization_grayscale))

    def _wire_view_actions(self):
        """Wire View menu actions for histogram display"""
        from processing.qt import show_input_histogram, show_output_histogram, show_input_output_histogram

        # Input Histogram
        act_input_hist = self.findChild(QAction, 'actionInput')
        if act_input_hist:
            act_input_hist.triggered.connect(self._show_input_histogram)

        # Output Histogram
        act_output_hist = self.findChild(QAction, 'actionOutput')
        if act_output_hist:
            act_output_hist.triggered.connect(self._show_output_histogram)

        # Input Output Histogram
        act_input_output_hist = self.findChild(QAction, 'actionInput_Output')
        if act_input_output_hist:
            act_input_output_hist.triggered.connect(self._show_input_output_histogram)

    def _show_input_histogram(self):
        """Show histogram for input image"""
        if not self._require_input():
            return
        from processing.qt import show_input_histogram
        show_input_histogram(self._input_pixmap)

    def _show_output_histogram(self):
        """Show histogram for output image"""
        from processing.qt import show_output_histogram
        show_output_histogram(self._output_pixmap)

    def _show_input_output_histogram(self):
        """Show histograms for both input and output images"""
        if not self._require_input():
            return
        from processing.qt import show_input_output_histogram
        show_input_output_histogram(self._input_pixmap, self._output_pixmap)

    def _wire_arithmetic_actions(self):
        """Wire Arithmetical Operation menu actions"""
        # Connect the menu itself to open the dialog
        arithmetic_menu = self.findChild(QMenu, 'menuAritmatical_Operation')
        if arithmetic_menu:
            arithmetic_menu.aboutToShow.connect(self._on_open_arithmetic_operations)

    def _wire_filter_actions(self):
        """Wire Filter menu actions"""
        from functools import partial
        from processing import ops

        # Filter mappings
        filter_mapping = {
            'actionIdentity': ops.identity,
            'actionSharpen': ops.sharpen,
            'actionUnsharp_Masking': ops.unsharp_masking,
            'actionAverage_Filter': ops.average_filter,
            'actionLow_Pass_Filter': ops.low_pass_filter,
            'actionHight_Pass_Filter': ops.high_pass_filter,
            'actionBandstop_Filter': ops.bandstop_filter,
            'actionEdge_Detection_1': ops.edge_detection_1,
            'actionEdge_Detection_2': ops.edge_detection_2,
            'actionEdge_Detection_3': ops.edge_detection_3,
            'actionGaussian_Blur_3x3': ops.gaussian_blur_3x3,
            'actionGaussian_Blur_5x5': ops.gaussian_blur_5x5,
        }

        for action_name, func in filter_mapping.items():
            act = self.findChild(QAction, action_name)
            if act:
                act.triggered.connect(partial(self._apply_and_show, func))

    def _wire_edge_detection_actions(self):
        """Wire standalone Edge Detection menu actions"""
        from functools import partial
        from processing import ops

        # Standalone Edge Detection mappings
        edge_detection_mapping = {
            'actionPrewitt': ops.prewitt,
            'actionSobel': ops.sobel,
        }

        for action_name, func in edge_detection_mapping.items():
            act = self.findChild(QAction, action_name)
            if act:
                act.triggered.connect(partial(self._apply_and_show, func))

    def _on_open_arithmetic_operations(self):
        """Open Arithmetic Operations dialog"""
        from ui.arithmetic_dialog import ArithmeticDialog
        # Pass the current input image if available, otherwise None
        input_image = self._input_pixmap if hasattr(self, '_input_pixmap') and self._input_pixmap else None
        dialog = ArithmeticDialog(self, input_image)
        dialog.exec_()

    def show_tentang(self):
        if self.tentang_window is None:
            self.tentang_window = TentangWindow(on_close=self._reset_tentang_window)
        self.tentang_window.show()
        self.tentang_window.raise_()
        self.tentang_window.activateWindow()

    def _reset_tentang_window(self):
        self.tentang_window = None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep the displayed image fitted when the window/view resizes
        if self._left_scene and not self._left_scene.items() == []:
            self.graphicsView.fitInView(self._left_scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        if self._right_scene and not self._right_scene.items() == []:
            self.graphicsView_2.fitInView(self._right_scene.itemsBoundingRect(), Qt.KeepAspectRatio)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
