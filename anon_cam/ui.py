"""PyQt UI for AnonCam."""

import logging
import time

import cv2
from PyQt6.QtCore import Qt, QSettings, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtMultimedia import QMediaDevices
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QSizePolicy,
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


def enumerate_cameras(max_index=10):
    names = []
    try:
        devices = QMediaDevices()
        names = [dev.description() for dev in devices.videoInputs()]
    except Exception:
        names = []
    result = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            if names and i < len(names):
                label = names[len(names) - 1 - i]
            else:
                label = f'Camera {i}'
            result.append((i, label))
    return result


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
        self.video.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Ignored,
        )

        self.mode_group = QButtonGroup(self)
        modes_row = QHBoxLayout()
        self.mode_buttons = {}
        for name, label in [
            ('auto', 'Auto'),
            ('faces', 'Faces'),
            ('all', 'All'),
            ('none', 'None')
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

        self.device_combo = QComboBox()
        self._refresh_device_combo()

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
        self.miss_thresh.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.recover_frames = QSpinBox()
        self.recover_frames.setRange(1, 10)
        self.recover_frames.setValue(2)
        self.recover_frames.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.det_every = QSpinBox()
        self.det_every.setRange(1, 10)
        self.det_every.setValue(3)
        self.det_every.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.det_width = QSpinBox()
        self.det_width.setRange(160, 960)
        self.det_width.setValue(480)
        self.det_width.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.expand = QSlider(Qt.Orientation.Horizontal)
        self.expand.setRange(0, 100)
        self.expand.setValue(20)

        self.grayscale = QCheckBox()
        self.mirror = QCheckBox()

        self.vcam_checkbox = QCheckBox()
        if pyvirtualcam is None:
            self.vcam_checkbox.setEnabled(False)
            self.vcam_checkbox.setToolTip('Requires the pyvirtualcam package')

        self.btn_start = QPushButton('Start')
        self.btn_stop = QPushButton('Stop')
        self.btn_logs = QPushButton('Logs')
        self.btn_reset = QPushButton('Reset settings')

        button_style = (
            "QPushButton {"
            " border-radius: 8px;"
            " padding: 6px 16px;"
            " background-color: #2b6df2;"
            " color: white;"
            " border: 1px solid #1f4fb3;"
            "}"
            "QPushButton:hover {"
            " background-color: #3a7bff;"
            "}"
            "QPushButton:pressed {"
            " background-color: #204fae;"
            "}"
            "QPushButton:disabled {"
            " background-color: #9aa9d6;"
            " border-color: #7f8bb5;"
            "}"
        )
        self.btn_start.setStyleSheet(button_style)
        self.btn_stop.setStyleSheet(button_style)
        self.btn_logs.setStyleSheet(button_style)
        self.btn_reset.setStyleSheet(button_style)

        self.hud = QLabel('')
        self.hud.setMinimumWidth(320)
        self.hud.setMinimumHeight(28)

        def style_form(f):
            f.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
            f.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            f.setHorizontalSpacing(20)

        gb_source = QGroupBox('Source')
        f_source = QFormLayout()
        f_source.addRow('Camera', self.device_combo)
        style_form(f_source)
        gb_source.setLayout(f_source)

        gb_mode = QGroupBox('Mode')
        f_mode = QFormLayout()
        f_mode.addRow(modes_wrap)
        style_form(f_mode)
        gb_mode.setLayout(f_mode)

        gb_anon = QGroupBox('Anonymization')
        f_anon = QFormLayout()
        f_anon.addRow('Strength', self.strength)
        f_anon.addRow('Feather', self.feather)
        f_anon.addRow('Mask expansion (%)', self.expand)
        f_anon.addRow('Only the largest', self.only_largest)
        style_form(f_anon)
        gb_anon.setLayout(f_anon)

        gb_det = QGroupBox('Detection')
        f_det = QFormLayout()
        f_det.addRow('Detection threshold (%)', self.conf)
        f_det.addRow('Det every (frames)', self.det_every)
        f_det.addRow('Det width (px)', self.det_width)
        style_form(f_det)
        gb_det.setLayout(f_det)

        gb_behavior = QGroupBox('Behavior')
        f_behavior = QFormLayout()
        f_behavior.addRow('Miss thresh (frames)', self.miss_thresh)
        f_behavior.addRow('Recover frames', self.recover_frames)
        style_form(f_behavior)
        gb_behavior.setLayout(f_behavior)

        gb_output = QGroupBox('Output')
        f_output = QFormLayout()
        f_output.addRow('B/W output', self.grayscale)
        f_output.addRow('Mirror image', self.mirror)
        f_output.addRow('Virtual camera output', self.vcam_checkbox)
        style_form(f_output)
        gb_output.setLayout(f_output)

        buttons = QHBoxLayout()
        buttons.addWidget(self.btn_start)
        buttons.addWidget(self.btn_stop)
        buttons.addWidget(self.btn_logs)
        buttons.addWidget(self.btn_reset)

        right = QVBoxLayout()
        right.addWidget(gb_source)
        right.addWidget(gb_mode)
        right.addWidget(gb_anon)
        right.addWidget(gb_det)
        right.addWidget(gb_behavior)
        right.addWidget(gb_output)
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
        self.setMinimumSize(900, 500)
        self.setMaximumSize(1680, 900)

        self.btn_start.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_logs.clicked.connect(self.show_logs)
        self.btn_reset.clicked.connect(self.reset_settings)
        self.grayscale.stateChanged.connect(self.on_grayscale_changed)
        self.vcam_checkbox.stateChanged.connect(self.on_vcam_toggled)
        self.device_combo.currentIndexChanged.connect(self.on_device_changed)

        for w in (
            self.miss_thresh, self.recover_frames, self.det_every,
            self.det_width, self.strength, self.feather, self.conf,
            self.expand,
        ):
            w.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._load_settings()

    def current_mode_value(self):
        btn = self.mode_group.checkedButton()
        val = btn.property('mode_value') if btn else None
        return val if isinstance(val, str) else 'auto'

    def _defaults(self):
        return {
            'mode': 'auto',
            'strength': 7,
            'feather': 20,
            'expand': 20,
            'only_largest': False,
            'conf': 50,
            'miss_thresh': 5,
            'recover_frames': 2,
            'det_every': 3,
            'det_width': 480,
            'grayscale': False,
            'mirror': False,
            'vcam_checked': False,
            'device_index': 0,
        }

    def _state_to_settings(self):
        return {
            'mode': self.current_mode_value(),
            'strength': self.strength.value(),
            'feather': self.feather.value(),
            'expand': self.expand.value(),
            'only_largest': self.only_largest.isChecked(),
            'conf': self.conf.value(),
            'miss_thresh': self.miss_thresh.value(),
            'recover_frames': self.recover_frames.value(),
            'det_every': self.det_every.value(),
            'det_width': self.det_width.value(),
            'grayscale': self.grayscale.isChecked(),
            'mirror': self.mirror.isChecked(),
            'vcam_checked': self.vcam_checkbox.isChecked(),
            'device_index': self.device_combo.currentData(),
        }

    def _apply_state(self, state):
        self.mode_buttons.get(state.get('mode', 'auto'), self.mode_buttons['auto']).setChecked(True)
        self.strength.setValue(state.get('strength', 7))
        self.feather.setValue(state.get('feather', 20))
        self.expand.setValue(state.get('expand', 20))
        self.only_largest.setChecked(state.get('only_largest', False))
        self.conf.setValue(state.get('conf', 50))
        self.miss_thresh.setValue(state.get('miss_thresh', 5))
        self.recover_frames.setValue(state.get('recover_frames', 2))
        self.det_every.setValue(state.get('det_every', 3))
        self.det_width.setValue(state.get('det_width', 480))
        self.grayscale.setChecked(state.get('grayscale', False))
        self.mirror.setChecked(state.get('mirror', False))
        self.vcam_checkbox.setChecked(state.get('vcam_checked', False))
        idx = state.get('device_index', 0)
        if idx is not None and idx >= 0:
            i = self.device_combo.findData(idx)
            if i >= 0:
                self.device_combo.blockSignals(True)
                self.device_combo.setCurrentIndex(i)
                self.device_combo.blockSignals(False)

    def _load_settings(self):
        s = QSettings('anon_cam', 'AnonCam')
        state = self._defaults()
        state['mode'] = s.value('mode', state['mode'], type=str)
        state['strength'] = s.value('strength', state['strength'], type=int)
        state['feather'] = s.value('feather', state['feather'], type=int)
        state['expand'] = s.value('expand', state['expand'], type=int)
        state['only_largest'] = s.value('only_largest', state['only_largest'], type=bool)
        state['conf'] = s.value('conf', state['conf'], type=int)
        state['miss_thresh'] = s.value('miss_thresh', state['miss_thresh'], type=int)
        state['recover_frames'] = s.value('recover_frames', state['recover_frames'], type=int)
        state['det_every'] = s.value('det_every', state['det_every'], type=int)
        state['det_width'] = s.value('det_width', state['det_width'], type=int)
        state['grayscale'] = s.value('grayscale', state['grayscale'], type=bool)
        state['mirror'] = s.value('mirror', state['mirror'], type=bool)
        state['vcam_checked'] = s.value('vcam_checked', state['vcam_checked'], type=bool)
        state['device_index'] = s.value('device_index', state['device_index'], type=int)
        self._apply_state(state)

    def _save_settings(self):
        s = QSettings('anon_cam', 'AnonCam')
        state = self._state_to_settings()
        for k, v in state.items():
            s.setValue(k, v)

    def reset_settings(self):
        self._apply_state(self._defaults())
        self._save_settings()
        self.logger.info('Settings reset to defaults')

    def showEvent(self, event):
        super().showEvent(event)
        self.btn_start.setFocus()

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

    def _refresh_device_combo(self):
        prev_data = self.device_combo.currentData() if self.device_combo.count() else None
        self.device_combo.clear()
        for idx, label in enumerate_cameras():
            self.device_combo.addItem(label, idx)
        if prev_data is not None:
            i = self.device_combo.findData(prev_data)
            if i >= 0:
                self.device_combo.setCurrentIndex(i)
        if self.device_combo.count() == 0:
            self.device_combo.addItem('No cameras', -1)

    def on_device_changed(self):
        if self.cap is not None:
            self.stop_camera()
            self.start_camera()

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
            mirror=bool(self.mirror.isChecked()),
            vcam_enabled=(pyvirtualcam is not None and self.vcam_checkbox.isChecked())
        )

    def start_camera(self):
        if self.cap is not None:
            self.logger.debug('Video capture already running')
            return
        index = self.device_combo.currentData()
        if index is None or index < 0:
            self.logger.warning('No camera selected')
            return
        self.cap = cv2.VideoCapture(int(index))
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
        self._save_settings()
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
        if cfg['mirror']:
            frame_to_show = cv2.flip(frame_to_show, 1)

        dt = time.time() - self.t0
        fps = 1.0 / dt if dt > 0 else 0.0
        self.fps_avg = self.fps_avg * 0.9 + fps * 0.1 if self.fps_avg > 0 else fps
        self.t0 = time.time()

        fh, fw = frame_to_show.shape[:2]
        lw, lh = self.video.width(), self.video.height()
        if lw > 0 and lh > 0:
            scale = min(lw / fw, lh / fh)
            nw, nh = int(fw * scale), int(fh * scale)
            nw, nh = max(1, nw), max(1, nh)
            scaled = cv2.resize(
                frame_to_show, (nw, nh), interpolation=cv2.INTER_LINEAR
            )
            qimg = QImage(
                scaled.data, nw, nh, nw * frame_to_show.shape[2],
                QImage.Format.Format_BGR888,
            ).copy()
        else:
            qimg = QImage(
                frame_to_show.data, fw, fh, fw * frame_to_show.shape[2],
                QImage.Format.Format_BGR888,
            ).copy()
        self.video.setPixmap(QPixmap.fromImage(qimg))

        txt = f"mode:{disp_mode} strength:{cfg['strength']} feather:{cfg['feather']} faces:{n_faces} FPS:{self.fps_avg:.1f}"
        self.hud.setText(txt)

        if cfg['vcam_enabled']:
            self._push_virtual_cam(frame_to_show, self.fps_avg if self.fps_avg > 0 else fps)
        else:
            self._close_vcam()
