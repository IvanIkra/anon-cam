"""Application entry point."""

import logging
import sys

from PyQt6.QtWidgets import QApplication

from .ui import AnonCamWindow


def main():
    app = QApplication(sys.argv)
    window = AnonCamWindow()
    window.resize(1280, 800)
    window.show()
    logging.getLogger('anon_cam').info('Application started')
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
