"""Qt logging helpers for the GUI."""

import logging

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import QMainWindow, QPlainTextEdit


class LogEmitter(QObject):
    message = pyqtSignal(str, int)


class QtLogHandler(logging.Handler):
    def __init__(self, history_limit: int = 500):
        super().__init__()
        self.emitter = LogEmitter()
        self.history: list[tuple[str, int]] = []
        self.history_limit = history_limit

    def emit(self, record: logging.LogRecord):
        msg = self.format(record)
        levelno = record.levelno
        self.history.append((msg, levelno))
        if len(self.history) > self.history_limit:
            self.history.pop(0)
        self.emitter.message.emit(msg, levelno)


def _color_for_level(levelno: int) -> QColor:
    if levelno >= logging.ERROR:
        return QColor(180, 50, 50)
    if levelno >= logging.WARNING:
        return QColor(180, 100, 0)
    if levelno >= logging.INFO:
        return QColor(30, 80, 120)
    return QColor(90, 90, 90)


class LogWindow(QMainWindow):
    def __init__(self, handler: QtLogHandler):
        super().__init__()
        self.setWindowTitle('AnonCam Logs')
        self.resize(600, 400)
        self._handler = handler
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.setCentralWidget(self.view)
        self._handler.emitter.message.connect(self.append_message)
        for line, levelno in self._handler.history:
            self.append_message(line, levelno, scroll=False)
        self.view.moveCursor(QTextCursor.MoveOperation.End)

    def append_message(self, message: str, levelno: int = logging.INFO, scroll: bool = True):
        cursor = self.view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(_color_for_level(levelno))
        cursor.mergeCharFormat(fmt)
        cursor.insertText(message + '\n')
        self.view.setTextCursor(cursor)
        if scroll:
            self.view.moveCursor(QTextCursor.MoveOperation.End)

    def closeEvent(self, event):
        event.ignore()
        self.hide()
