"""
config.py
- Central place to keep global thresholds and device selection.
"""

import torch

# Confidence threshold for detections. Boxes with lower conf will be ignored.
CONF_THRESH = 0.90

# IoU threshold used during NMS to merge overlapping boxes.
IOU_THRESH  = 0.50

def get_device():
    """
    Decide which device to use:
      - 'mps' on Apple Silicon if available (faster than CPU)
      - otherwise fall back to 'cpu'
      - Ultralytics can also auto-select, but we expose this so our code stays explicit and testable.
    """
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
