"""PyQt UI for AnonCam."""

import logging
import time

import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .engine import AnonEngine
from .logging_utils import LogWindow, QtLogHandler

try:
    import pyvirtualcam
    from pyvirtualcam import PixelFormat
except ImportError:  # optional dependency
    pyvirtualcam = None
    PixelFormat = None


class AnonCamWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('AnonCam')
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_timer)
        self.engine = AnonEngine()
        self.t0 = time.time()
        self.fps_avg = 0.0
        self.vcam = None
        self.log_window = None

        self.logger = logging.getLogger('anon_cam')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.log_handler = QtLogHandler()
        self.log_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
        if not any(isinstance(h, QtLogHandler) for h in self.logger.handlers):
            self.logger.addHandler(self.log_handler)

        self.video = QLabel()
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.mode_group = QButtonGroup(self)
        modes_row = QHBoxLayout()
        self.mode_buttons = {}
        for name, label in [
            ('auto', 'Авто'),
            ('faces', 'Лица'),
            ('all', 'Все'),
            ('none', 'Нет')
        ]:
            btn = QRadioButton(label)
            self.mode_group.addButton(btn)
            btn.setProperty('mode_value', name)
            modes_row.addWidget(btn)
            self.mode_buttons[name] = btn
        self.mode_buttons['auto'].setChecked(True)
        modes_row.addStretch(1)
        modes_wrap = QWidget()
        modes_wrap.setLayout(modes_row)

        self.strength = QSlider(Qt.Orientation.Horizontal)
        self.strength.setRange(1, 10)
        self.strength.setValue(7)

        self.feather = QSlider(Qt.Orientation.Horizontal)
        self.feather.setRange(0, 60)
        self.feather.setValue(20)

        self.only_largest = QCheckBox()

        self.conf = QSlider(Qt.Orientation.Horizontal)
        self.conf.setRange(0, 100)
        self.conf.setValue(50)

        self.miss_thresh = QSpinBox()
        self.miss_thresh.setRange(1, 60)
        self.miss_thresh.setValue(5)

        self.recover_frames = QSpinBox()
        self.recover_frames.setRange(1, 10)
        self.recover_frames.setValue(2)

        self.det_every = QSpinBox()
        self.det_every.setRange(1, 10)
        self.det_every.setValue(3)

        self.det_width = QSpinBox()
        self.det_width.setRange(160, 960)
        self.det_width.setValue(480)

        self.expand = QSlider(Qt.Orientation.Horizontal)
        self.expand.setRange(0, 100)
        self.expand.setValue(20)

        self.grayscale = QCheckBox()

        self.vcam_checkbox = QCheckBox('Вывод в виртуальную камеру')
        if pyvirtualcam is None:
            self.vcam_checkbox.setEnabled(False)
            self.vcam_checkbox.setToolTip('Требуется пакет pyvirtualcam')

        self.btn_start = QPushButton('Старт')
        self.btn_stop = QPushButton('Стоп')
        self.btn_logs = QPushButton('Логи')

        self.hud = QLabel('')

        controls = QGroupBox('Настройки')
        form = QFormLayout()
        form.addRow('Режим', modes_wrap)
        form.addRow('Сила', self.strength)
        form.addRow('Перо', self.feather)
        form.addRow('Только крупнейшее', self.only_largest)
        form.addRow('Порог детекции (%)', self.conf)
        form.addRow('Miss thresh (кадры)', self.miss_thresh)
        form.addRow('Recover frames', self.recover_frames)
        form.addRow('Det every (кадры)', self.det_every)
        form.addRow('Det width (px)', self.det_width)
        form.addRow('Увеличение маски (%)', self.expand)
        form.addRow('Ч/Б вывод', self.grayscale)
        form.addRow(self.vcam_checkbox)
        controls.setLayout(form)

        buttons = QHBoxLayout()
        buttons.addWidget(self.btn_start)
        buttons.addWidget(self.btn_stop)
        buttons.addWidget(self.btn_logs)

        right = QVBoxLayout()
        right.addWidget(controls)
        right.addLayout(buttons)
        right.addWidget(self.hud)
        right.addStretch()

        root = QHBoxLayout()
        root.addWidget(self.video, 3)
        wrap = QWidget()
        wrap.setLayout(right)
        root.addWidget(wrap, 1)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

        self.btn_start.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_logs.clicked.connect(self.show_logs)
        self.grayscale.stateChanged.connect(self.on_grayscale_changed)
        self.vcam_checkbox.stateChanged.connect(self.on_vcam_toggled)

    def current_mode_value(self):
        btn = self.mode_group.checkedButton()
        val = btn.property('mode_value') if btn else None
        return val if isinstance(val, str) else 'auto'

    def show_logs(self):
        if self.log_window is None:
            self.log_window = LogWindow(self.log_handler)
        self.log_window.show()
        self.log_window.raise_()
        self.log_window.activateWindow()
        self.logger.info('Log window opened')

    def on_grayscale_changed(self, state):
        enabled = state == Qt.CheckState.Checked
        self.logger.info('Grayscale output enabled' if enabled else 'Grayscale output disabled')

    def on_vcam_toggled(self, state):
        enabled = state == Qt.CheckState.Checked
        self.logger.info('Virtual camera enabled' if enabled else 'Virtual camera disabled')

    def build_cfg(self):
        return dict(
            mode=self.current_mode_value(),
            strength=int(self.strength.value()),
            feather=int(self.feather.value()),
            only_largest=bool(self.only_largest.isChecked()),
            conf=float(self.conf.value()) / 100.0,
            miss_thresh=int(self.miss_thresh.value()),
            recover_frames=int(self.recover_frames.value()),
            det_every=int(self.det_every.value()),
            det_width=int(self.det_width.value()),
            expand=float(self.expand.value()) / 100.0,
            grayscale=bool(self.grayscale.isChecked()),
            vcam_enabled=(pyvirtualcam is not None and self.vcam_checkbox.isChecked())
        )

    def start_camera(self):
        if self.cap is not None:
            self.logger.debug('Video capture already running')
            return
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.timer.start(0)
        self.logger.info('Video capture started')

    def stop_camera(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.logger.info('Video capture stopped')
        self.video.clear()
        self.hud.setText('')
        self._close_vcam()

    def closeEvent(self, event):
        self.stop_camera()
        return super().closeEvent(event)

    def _close_vcam(self):
        if self.vcam is not None:
            try:
                self.vcam.close()
            except Exception:
                pass
            else:
                self.logger.info('Virtual camera closed')
            self.vcam = None

    def _push_virtual_cam(self, frame, fps_hint):
        if pyvirtualcam is None:
            return
        h, w = frame.shape[:2]
        target_fps = max(1, int(fps_hint)) if fps_hint > 0 else 30
        if self.vcam is None or self.vcam.width != w or self.vcam.height != h:
            self._close_vcam()
            try:
                self.vcam = pyvirtualcam.Camera(width=w, height=h, fps=target_fps, fmt=PixelFormat.BGR)
                self.logger.info(f'Virtual camera opened {w}x{h}@{target_fps}fps')
            except Exception as exc:
                self.hud.setText(f'VCam error: {exc}')
                self.vcam_checkbox.setChecked(False)
                self.logger.error(f'Virtual camera error: {exc}')
                self._close_vcam()
                return
        try:
            self.vcam.send(frame)
            self.vcam.sleep_until_next_frame()
        except Exception as exc:
            self.hud.setText(f'VCam error: {exc}')
            self.vcam_checkbox.setChecked(False)
            self.logger.error(f'Virtual camera error: {exc}')
            self._close_vcam()

    def on_timer(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.logger.warning('Failed to read frame from camera; stopping capture')
            self.stop_camera()
            return

        cfg = self.build_cfg()
        out, n_faces, disp_mode = self.engine.process(frame, cfg)

        frame_to_show = out
        if cfg['grayscale']:
            gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            frame_to_show = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        dt = time.time() - self.t0
        fps = 1.0 / dt if dt > 0 else 0.0
        self.fps_avg = self.fps_avg * 0.9 + fps * 0.1 if self.fps_avg > 0 else fps
        self.t0 = time.time()

        h, w, ch = frame_to_show.shape
        qimg = QImage(frame_to_show.data, w, h, ch * w, QImage.Format.Format_BGR888)
        self.video.setPixmap(QPixmap.fromImage(qimg))

        txt = f"mode:{disp_mode} strength:{cfg['strength']} feather:{cfg['feather']} faces:{n_faces} FPS:{self.fps_avg:.1f}"
        self.hud.setText(txt)

        if cfg['vcam_enabled']:
            self._push_virtual_cam(frame_to_show, self.fps_avg if self.fps_avg > 0 else fps)
        else:
            self._close_vcam()
