"""Face detection utilities."""

import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(self, conf: float = 0.5):
        self.conf = conf
        self._mp = mp.solutions.face_detection
        self.det = self._mp.FaceDetection(model_selection=1, min_detection_confidence=0.3)

    def detect(self, frame_rgb, det_width: int, conf_thresh: float):
        h, w = frame_rgb.shape[:2]
        scale = det_width / float(w)
        if det_width < w:
            small = self._resize(frame_rgb, det_width, h, scale)
        else:
            small = frame_rgb
            scale = 1.0
        res = self.det.process(small)
        boxes, confs = [], []
        if res and res.detections:
            for detection in res.detections:
                score = detection.score[0] if detection.score else 0.0
                if score < conf_thresh:
                    continue
                bb = detection.location_data.relative_bounding_box
                x = bb.xmin * small.shape[1]
                y = bb.ymin * small.shape[0]
                w0 = bb.width * small.shape[1]
                h0 = bb.height * small.shape[0]
                x1 = int(x / scale)
                y1 = int(y / scale)
                x2 = int((x + w0) / scale)
                y2 = int((y + h0) / scale)
                boxes.append([x1, y1, x2, y2])
                confs.append(float(score))
        return boxes, confs

    @staticmethod
    def _resize(frame_rgb, det_width, h, scale):
        return cv2.resize(frame_rgb, (det_width, int(h * scale)), interpolation=cv2.INTER_AREA)
