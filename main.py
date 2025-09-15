import sys
import os

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QAction, QMenu
from PyQt5 import uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QGraphicsScene, QInputDialog
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtCore import Qt, QRectF

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(BASE_DIR,'ui')

class TentangWindow(QWidget):
    def __init__(self, parent=None, on_close=None):
        super().__init__(parent)
        uic.loadUi(os.path.join(UI_DIR, 'TentangWindow.ui'), self)
        self.on_close = on_close

    def closeEvent(self, event):
        if self.on_close:
            self.on_close()
        event.accept()

class MainWindow(QMainWindow):
    def __int__(self):
        super().__init__()
        uic.loadUi(os.path.join(UI_DIR, 'MainWindow.ui'), self)

        self._left_scene = QGraphicsScene(self.graphicsView)
        self.graphicsView.setScane(self._left_scene)
        self.graphicsView.setRenderHints(self.graphicsView.renderHints() |
                                         QPainter.Antialiasing |
                                         QPainter.SmoothPixmapTransform)

        self._right_scene = QGraphicsScene(self.graphicsView_2)
        self.graphicsView_2.setScane(self._right_scene)
        self.graphicsView_2.setRenderHints(self.graphicsView_2.renderHints() |
                                           QPainter.Antialiasing |
                                           QPainter.SmoothPixmapTransform)

        # Internal state
        self._input_pixamp: QPixmap = None
        self._output_pixamp: QPixmap = None

        buka_action = self.findChild(QAction, 'actionbuka')
        if buka_action is not None:
            buka_action.triggered.connect(self.open_image)

        simpan_sebagai = self.findChild(QAction, 'actionSimpan_Sebagai')
        if simpan_sebagai is not None:
            simpan_sebagai.triggered.connect(self.save_output_as)

        keluar_action = self.findChild(QAction, 'actionkeluar')
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

