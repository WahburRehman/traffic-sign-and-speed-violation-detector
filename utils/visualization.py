"""
visualization.py
- Small, reusable drawing helpers.
- Keep visualization separate from detection & business logic.
"""

import cv2, re
from typing import Tuple

def draw_box_with_label(frame, xyxy, label: str, conf: float,
                        color: Tuple[int,int,int]=(0, 200, 255)):
    """
    Draw a rectangle with a filled label box on top.

    frame: BGR image (numpy array)
    xyxy : [x1, y1, x2, y2] in pixels
    label: class name to show
    conf : confidence score to display
    color: box and label background color (B,G,R)
    """
    x1, y1, x2, y2 = map(int, xyxy)

    # 1) Draw rectangle (the bounding box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # 2) Build label text like "speed_limit_50 0.87"
    txt = f"{label} {conf:.2f}"

    # 3) Compute text box size to draw a filled background
    t_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.rectangle(frame, (x1, y1 - t_size[1] - 6), (x1 + t_size[0] + 6, y1), color, -1)

    # 4) Put text over the filled rectangle
    cv2.putText(frame, txt, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def overlay_top_banner(frame, message: str, status: str = "ok"):
    """
    Draw a top status banner (e.g., 'VIOLATION' or 'OK').

    status: 'ok' -> green text, 'bad' -> red text
    """
    h, w = frame.shape[:2]
    banner_h = 40
    bg = (0, 0, 0)
    color_ok  = (0, 180, 0)
    color_bad = (0, 0, 255)
    color = color_bad if status == "bad" else color_ok

    # 1) Black strip at top for readability
    cv2.rectangle(frame, (0, 0), (w, banner_h + 6), bg, -1)

    # 2) Status message in chosen color
    cv2.putText(frame, message, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    

# extract 30/50/70 from class name
num_pat = re.compile(r"(\d+)")         
def extract_speed_from_name(name):
    m = num_pat.search(name or "")
    return int(m.group(1)) if m else None
