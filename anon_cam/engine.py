"""Core anonymization engine."""

import logging

import cv2
import numpy as np

from .anonymize import irreversible_anonymize, draw_square_mask, blend_roi, select_boxes
from .detector import FaceDetector


class AnonEngine:
    def __init__(self):
        self.detector = FaceDetector()
        self.last_boxes = []
        self.frame_id = 0
        self.miss_count = 0
        self.found_count = 0
        self.prev_faces_state = None
        self.prev_display_mode = None
        self.logger = logging.getLogger('anon_cam')

    def process(self, frame, cfg):
        H, W = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        run_det = (self.frame_id % max(1, cfg['det_every']) == 0)
        boxes = list(self.last_boxes)

        if run_det:
            det_boxes, _ = self.detector.detect(rgb, cfg['det_width'], cfg['conf'])
            if det_boxes:
                boxes = select_boxes(det_boxes, cfg['only_largest'], W, H, cfg['expand'])
                self.last_boxes = boxes
                self.miss_count = 0
                self.found_count = min(cfg['recover_frames'], self.found_count + 1)
            else:
                self.miss_count += 1
                self.found_count = 0
                if self.miss_count >= cfg['miss_thresh']:
                    self.last_boxes = []
                    boxes = []

        faces_present = len(boxes) > 0
        if faces_present != self.prev_faces_state:
            self.logger.info('Faces detected' if faces_present else 'Faces lost')
            self.prev_faces_state = faces_present

        display_mode = cfg['mode']
        if cfg['mode'] == 'auto':
            display_mode = 'faces'
            if not faces_present and self.miss_count >= cfg['miss_thresh']:
                display_mode = 'all'
            if faces_present and self.found_count >= cfg['recover_frames']:
                display_mode = 'faces'

        if display_mode == 'faces':
            out = self._process_faces(frame, boxes, cfg['feather'], cfg['strength'])
        elif display_mode == 'all':
            out = irreversible_anonymize(frame, cfg['strength'])
        elif display_mode == 'none':
            out = frame
        else:
            out = self._process_faces(frame, boxes, cfg['feather'], cfg['strength'])

        if display_mode != self.prev_display_mode:
            self.logger.info(f'Display mode set to: {display_mode}')
            self.prev_display_mode = display_mode

        self.frame_id += 1
        return out, len(boxes), display_mode

    def _process_faces(self, frame, boxes, feather, strength):
        H, W = frame.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        anon = frame.copy()
        for box in boxes:
            x1, y1, x2, y2 = box
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            aroi = irreversible_anonymize(roi, strength)
            anon[y1:y2, x1:x2] = aroi
            mask = draw_square_mask(mask, box, feather)
        out = blend_roi(frame, anon, mask)
        return out
