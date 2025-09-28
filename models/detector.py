"""
detector.py
- Wrap Ultralytics YOLO in a small class with a clean .predict(frame) API.
- Returns a list of Detection objects.
"""

from dataclasses import dataclass
from typing import List
import numpy as np
from ultralytics import YOLO

@dataclass
class Detection:
    """
    A single detection result.

    xyxy: [x1, y1, x2, y2] pixel coordinates in the current frame
    conf: confidence score (0..1)
    cls_id: integer class index
    cls_name: human-readable class name (from model.names), or str(cls_id) if unknown
    """
    xyxy: np.ndarray
    conf: float
    cls_id: int
    cls_name: str

class YoloDetector:
    """
    Simple detector:
      - load a YOLO .pt model
      - run inference on a BGR frame (numpy array)
      - return structured detections
    """

    def __init__(self, model_path: str, device: str = "cpu", conf: float = 0.35, iou: float = 0.50):
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.iou = iou

        # Load YOLO model from disk (.pt file). Ultralytics handles weights & architecture.
        self.model = YOLO(self.model_path)

        # Try to move underlying torch model to target device (mps/cpu).
        # Ultralytics usually handles device, but we keep it explicit.
        try:
            self.model.model.to(self.device)
        except Exception:
            # If move fails (rare), continue; Ultralytics will still run on default device.
            pass

        # Class name mapping (e.g., 0 -> 'speed_limit_20', ...). Used for readable labels.
        self.names = self.model.names if hasattr(self.model, "names") else {}

    def predict(self, frame) -> List[Detection]:
        """
        Run inference on one BGR frame (numpy array).
        Returns a list of Detection objects.
        """
        # Ultralytics returns a list of results; we take the first result for this frame.
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False
        )
        dets: List[Detection] = []
        r0 = results[0]

        # If no boxes found, return empty list.
        if r0.boxes is None or len(r0.boxes) == 0:
            return dets

        # Extract tensors and convert to CPU numpy for consistency.
        boxes = r0.boxes
        xyxy = boxes.xyxy.detach().cpu().numpy()  # shape: (N, 4)
        conf = boxes.conf.detach().cpu().numpy()  # shape: (N,)
        cls  = boxes.cls.detach().cpu().numpy().astype(int)  # shape: (N,)

        # Build Detection dataclasses
        for i in range(len(boxes)):
            cid = int(cls[i])
            cname = self.names.get(cid, str(cid))
            dets.append(
                Detection(
                    xyxy=xyxy[i],
                    conf=float(conf[i]),
                    cls_id=cid,
                    cls_name=cname
                )
            )
        return dets
