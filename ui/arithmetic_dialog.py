import os
from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox, QGraphicsScene
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtCore import Qt, QRectF
from PyQt5 import uic
from processing.qt import pixmap_to_numpy, numpy_to_pixmap
from processing import ops

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI_DIR = os.path.join(BASE_DIR, 'ui')


class ArithmeticDialog(QDialog):
    def __init__(self, parent=None, input_image=None):
        super().__init__(parent)
        uic.loadUi(os.path.join(UI_DIR, 'ArithmeticDialog.ui'), self)

        # Setup graphics views
        self._input1_scene = QGraphicsScene(self.graphicsViewInput1)
        self.graphicsViewInput1.setScene(self._input1_scene)
        self.graphicsViewInput1.setRenderHints(self.graphicsViewInput1.renderHints() |
                                              QPainter.Antialiasing |
                                              QPainter.SmoothPixmapTransform)

        self._input2_scene = QGraphicsScene(self.graphicsViewInput2)
        self.graphicsViewInput2.setScene(self._input2_scene)
        self.graphicsViewInput2.setRenderHints(self.graphicsViewInput2.renderHints() |
                                              QPainter.Antialiasing |
                                              QPainter.SmoothPixmapTransform)

        self._output_scene = QGraphicsScene(self.graphicsViewOutput)
        self.graphicsViewOutput.setScene(self._output_scene)
        self.graphicsViewOutput.setRenderHints(self.graphicsViewOutput.renderHints() |
                                              QPainter.Antialiasing |
                                              QPainter.SmoothPixmapTransform)

        # Internal state
        self._input1_pixmap = input_image
        self._input2_pixmap = None
        self._output_pixmap = None

        # Load initial image if provided
        if input_image:
            self._display_pixmap_on_input1(input_image)

        # Connect signals
        self.pushButtonLoadInput1.clicked.connect(self._load_input1)
        self.pushButtonLoadInput2.clicked.connect(self._load_input2)
        self.pushButtonSaveOutput.clicked.connect(self._save_output)
        self.pushButtonExecute.clicked.connect(self._execute_operation)
        self.comboBoxOperation.currentTextChanged.connect(self._on_operation_changed)
        self.comboBoxType.currentTextChanged.connect(self._on_type_changed)

        # Initialize UI state
        self._on_operation_changed()
        self._on_type_changed()

    def _display_pixmap_on_input1(self, pixmap: QPixmap):
        """Display pixmap on input 1 graphics view"""
        if pixmap.isNull():
            return
        self._input1_scene.clear()
        item = self._input1_scene.addPixmap(pixmap)
        self._input1_scene.setSceneRect(QRectF(pixmap.rect()))
        self.graphicsViewInput1.fitInView(item, Qt.KeepAspectRatio)

    def _display_pixmap_on_input2(self, pixmap: QPixmap):
        """Display pixmap on input 2 graphics view"""
        if pixmap.isNull():
            return
        self._input2_scene.clear()
        item = self._input2_scene.addPixmap(pixmap)
        self._input2_scene.setSceneRect(QRectF(pixmap.rect()))
        self.graphicsViewInput2.fitInView(item, Qt.KeepAspectRatio)

    def _display_pixmap_on_output(self, pixmap: QPixmap):
        """Display pixmap on output graphics view"""
        if pixmap.isNull():
            return
        self._output_scene.clear()
        item = self._output_scene.addPixmap(pixmap)
        self._output_scene.setSceneRect(QRectF(pixmap.rect()))
        self.graphicsViewOutput.fitInView(item, Qt.KeepAspectRatio)

    def _load_input1(self):
        """Load image for input 1"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            'Load Input 1 Image',
            '',
            'Images (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff);;All Files (*)'
        )
        if not file_path:
            return
        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            QMessageBox.warning(self, 'Error', 'Cannot load the selected image.')
            return
        self._input1_pixmap = pixmap
        self._display_pixmap_on_input1(pixmap)

    def _load_input2(self):
        """Load image for input 2"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            'Load Input 2 Image',
            '',
            'Images (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff);;All Files (*)'
        )
        if not file_path:
            return
        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            QMessageBox.warning(self, 'Error', 'Cannot load the selected image.')
            return
        self._input2_pixmap = pixmap
        self._display_pixmap_on_input2(pixmap)

    def _save_output(self):
        """Save output image"""
        if self._output_pixmap is None or self._output_pixmap.isNull():
            QMessageBox.information(self, 'Save', 'No output image to save.')
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            'Save Output Image',
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
                file_path += '.png'  # Default to PNG if no extension

        try:
            if self._output_pixmap.save(file_path):
                QMessageBox.information(self, 'Save', f'Image saved successfully to:\n{file_path}')
            else:
                QMessageBox.warning(self, 'Save', f'Failed to save the image to:\n{file_path}\n\nPlease check file permissions and try again.')
        except Exception as e:
            QMessageBox.warning(self, 'Save', f'Error saving image:\n{str(e)}')

    def _on_operation_changed(self):
        """Handle operation combo box change"""
        operation = self.comboBoxOperation.currentText()
        if operation == "Blend":
            self.comboBoxType.setCurrentText("Image + Image")
            self.comboBoxType.setEnabled(False)
            self.doubleSpinBoxAlpha.setEnabled(True)
            self.doubleSpinBoxBeta.setEnabled(True)
            self.doubleSpinBoxConstant.setEnabled(False)
        else:
            self.comboBoxType.setEnabled(True)
            self.doubleSpinBoxAlpha.setEnabled(False)
            self.doubleSpinBoxBeta.setEnabled(False)
            self._on_type_changed()

    def _on_type_changed(self):
        """Handle type combo box change"""
        operation_type = self.comboBoxType.currentText()
        if operation_type == "Image + Constant":
            self.doubleSpinBoxConstant.setEnabled(True)
        else:
            self.doubleSpinBoxConstant.setEnabled(False)

    def _execute_operation(self):
        """Execute the selected arithmetic operation"""
        if self._input1_pixmap is None:
            QMessageBox.warning(self, 'Error', 'Please load Input 1 image.')
            return

        operation = self.comboBoxOperation.currentText()
        operation_type = self.comboBoxType.currentText()

        try:
            if operation_type == "Image + Image":
                if self._input2_pixmap is None:
                    QMessageBox.warning(self, 'Error', 'Please load Input 2 image for image-to-image operations.')
                    return

                input1_arr = pixmap_to_numpy(self._input1_pixmap)
                input2_arr = pixmap_to_numpy(self._input2_pixmap)

                if operation == "Add":
                    result = ops.add_images(input1_arr, input2_arr)
                elif operation == "Subtract":
                    result = ops.subtract_images(input1_arr, input2_arr)
                elif operation == "Multiply":
                    result = ops.multiply_images(input1_arr, input2_arr)
                elif operation == "Divide":
                    result = ops.divide_images(input1_arr, input2_arr)
                elif operation == "Absolute Difference":
                    result = ops.absolute_difference(input1_arr, input2_arr)
                elif operation == "Blend":
                    alpha = self.doubleSpinBoxAlpha.value()
                    beta = self.doubleSpinBoxBeta.value()
                    result = ops.blend_images(input1_arr, input2_arr, alpha, beta)

            else:  # Image + Constant
                input1_arr = pixmap_to_numpy(self._input1_pixmap)
                constant = self.doubleSpinBoxConstant.value()

                if operation == "Add":
                    result = ops.add_constant(input1_arr, constant)
                elif operation == "Subtract":
                    result = ops.subtract_constant(input1_arr, constant)
                elif operation == "Multiply":
                    result = ops.multiply_constant(input1_arr, constant)
                elif operation == "Divide":
                    result = ops.divide_constant(input1_arr, constant)

            # Convert result back to pixmap and display
            result_pixmap = numpy_to_pixmap(result)
            self._output_pixmap = result_pixmap
            self._display_pixmap_on_output(result_pixmap)

        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Operation failed: {str(e)}')

    def resizeEvent(self, event):
        """Handle window resize to keep images fitted"""
        super().resizeEvent(event)
        # Keep the displayed images fitted when the window resizes
        if self._input1_scene and not self._input1_scene.items() == []:
            self.graphicsViewInput1.fitInView(self._input1_scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        if self._input2_scene and not self._input2_scene.items() == []:
            self.graphicsViewInput2.fitInView(self._input2_scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        if self._output_scene and not self._output_scene.items() == []:
            self.graphicsViewOutput.fitInView(self._output_scene.itemsBoundingRect(), Qt.KeepAspectRatio)