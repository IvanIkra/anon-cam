"""Qt logging helpers for the GUI."""

import logging

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import QMainWindow, QPlainTextEdit


class LogEmitter(QObject):
    message = pyqtSignal(str)


class QtLogHandler(logging.Handler):
    def __init__(self, history_limit: int = 500):
        super().__init__()
        self.emitter = LogEmitter()
        self.history: list[str] = []
        self.history_limit = history_limit

    def emit(self, record: logging.LogRecord):
        msg = self.format(record)
        self.history.append(msg)
        if len(self.history) > self.history_limit:
            self.history.pop(0)
        self.emitter.message.emit(msg)


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
        for line in self._handler.history:
            self.append_message(line, scroll=False)
        self.view.moveCursor(QTextCursor.MoveOperation.End)

    def append_message(self, message: str, scroll: bool = True):
        self.view.appendPlainText(message)
        if scroll:
            cursor = self.view.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.view.setTextCursor(cursor)

    def closeEvent(self, event):
        event.ignore()
        self.hide()
