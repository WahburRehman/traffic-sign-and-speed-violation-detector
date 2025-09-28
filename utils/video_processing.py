"""
Reusable video processing pipeline for Traffic Sign Detection.
- Loads YOLO model once
- Iterates frames, draws boxes, applies banner logic
- Writes an annotated MP4
- Returns simple KPIs and an events list for CSV
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple
from pathlib import Path

import cv2, time, csv, tempfile

from utils import draw_box_with_label, eval_rule, overlay_top_banner
from config import get_device, CONF_THRESH, IOU_THRESH
from models import YoloDetector


# -------- Types --------
Event = Dict[str, object]
Kpis = Dict[str, object]
ProgressCb = Callable[[int, int, float], None]  # (processed_frames, total_frames, fps_estimate)

# -------- Constants --------
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def _tiny_box_filter(xyxy: Tuple[float, float, float, float], min_area: float) -> bool:
    """Return True if bbox area >= min_area else False."""
    x1, y1, x2, y2 = map(int, xyxy)
    return (x2 - x1) * (y2 - y1) >= min_area

def process_video(
    video_path: str,
    model_path: str,
    user_speed_kmh: float,
    output_path: str,
    min_box_area_ratio: float = 0.0002,  # 0.02% of frame
    progress_cb: Optional[ProgressCb] = None,
) -> Tuple[Kpis, List[Event]]:
    """
    Core pipeline:
      - Open video, run detection frame-by-frame
      - Draw boxes + highest-priority banner per frame
      - Save annotated MP4
      - Collect events for CSV (timecode, class, conf, bbox, status)
      - Return KPIs + events

    Args:
        video_path: input video file
        model_path: path to YOLO .pt weights
        user_speed_kmh: current vehicle speed to evaluate rules
        output_path: where to save annotated MP4
        min_box_area_ratio: skip tiny boxes (relative to frame area)
        progress_cb: optional callback(processed, total, fps_est) for UI

    Returns:
        kpis: dict with totals etc.
        events: list of dict rows for CSV
    """
    # ---- Setup IO ----
    in_path = Path(video_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    width_even = width - (width % 2)
    height_even = height - (height % 2)
    needs_crop = (width_even != width) or (height_even != height)

    writer = None
    for cc in ("avc1", "mp4v", "XVID"):
        w = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*cc), fps, (width_even, height_even))
        if w.isOpened():
            writer = w
            used_codec = cc
            break

    if writer is None:
        cap.release()
        raise RuntimeError(f"Could not open writer for: {out_path} with avc1/mp4v/XVID")

    # ---- Model ----
    device = get_device()
    detector = YoloDetector(model_path, device=device, conf=CONF_THRESH, iou=IOU_THRESH)

    # ---- Thresholds ----
    frame_area = width * height
    min_box_area = frame_area * min_box_area_ratio

    # ---- Stats / outputs ----
    total_detections = 0
    total_violations = 0
    top_class_counts: Dict[str, int] = {}
    events: List[Event] = []

    # Simple FPS estimate (updated gradually)
    t0 = time.time()
    processed = 0
    fps_est = 0.0

    # ---- Main loop ----
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        best_banner = None  # (status, text, priority, conf)
        # Run inference
        detections = detector.predict(frame)

        for d in detections:
            # Skip tiny boxes if any
            if not _tiny_box_filter(d.xyxy, min_box_area):
                continue

            total_detections += 1
            # Draw all detections (visual evidence)
            draw_box_with_label(frame, d.xyxy, d.cls_name, d.conf)

            # Evaluate rule for banner (e.g., speed limit check)
            rule_evaluation = eval_rule(d.cls_name, {"user_speed": float(user_speed_kmh)})
            # If no rule defined, we still record the detection event
            status = None
            text = None
            priority = -1

            if rule_evaluation is not None:
                status, text, priority = rule_evaluation
                # Track highest-priority banner per frame (tie-break by conf)
                if (best_banner is None) or (priority > best_banner[2]) or (
                    priority == best_banner[2] and d.conf > best_banner[3]
                ):
                    best_banner = (status, text, priority, d.conf)

                # Violation counting (simple: status == "violation")
                if status == "violation":
                    total_violations += 1

            # Record an event row for CSV
            x1, y1, x2, y2 = map(int, d.xyxy)
            timecode_s = (processed / fps) if fps > 0 else None
            events.append(
                {
                    "frame": processed,
                    "time_s": round(timecode_s, 3) if timecode_s is not None else None,
                    "class": d.cls_name,
                    "conf": float(d.conf),
                    "bbox": [x1, y1, x2, y2],
                    "status": status,       # None | "ok" | "warning" | "violation"
                    "banner_text": text,    # None or human-readable string
                }
            )

            # Count class frequency for a simple "top-1 violation type"
            top_class_counts[d.cls_name] = top_class_counts.get(d.cls_name, 0) + 1

        # Overlay a single banner if selected for this frame
        if best_banner is not None:
            overlay_top_banner(frame, best_banner[1], status=best_banner[0])

        # Ensure frame matches writer size if original was odd-sized
        if needs_crop:
            frame = frame[:height_even, :width_even]
        # Write annotated frame
        writer.write(frame)

        # Progress & FPS estimate
        processed += 1
        if processed % 10 == 0:
            dt = time.time() - t0
            fps_est = processed / dt if dt > 0 else 0.0
            if progress_cb:
                progress_cb(processed, total_frames, fps_est)

    # Cleanup
    cap.release()
    writer.release()

    # Compute KPIs
    top1_class = None
    if top_class_counts:
        top1_class = max(top_class_counts.items(), key=lambda kv: kv[1])[0]

    kpis: Kpis = {
        "total_detections": total_detections,
        "total_violations": total_violations,
        "top_class": top1_class,
        "fps_estimate": round(fps_est, 2),
        "frames": processed,
        "output_path": str(out_path),
    }
    return kpis, events


def save_events_csv(events: List[Event], csv_path: str) -> None:
    """Write events list to a CSV with stable columns."""
    cols = ["frame", "time_s", "class", "conf", "bbox", "status", "banner_text"]
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for e in events:
            row = {k: e.get(k) for k in cols}
            w.writerow(row)


# -------- FOLLOWING FUNCTIONS ARE MAINLY USED IN APP.py FOR STREAMTLIT --------

def get_sample_videos(samples_dir: Path) -> list[Path]:
    if not samples_dir.exists():
        return []
    return sorted([p for p in samples_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS])


def get_video_meta(path: Path) -> dict:
    """Read basic meta using OpenCV."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    dur = (fc / fps) if fps > 0 else None
    return {
        "fps": round(float(fps), 2) if fps else None,
        "frames": fc if fc else None,
        "resolution": f"{w}x{h}" if w and h else None,
        "duration_sec": round(dur, 2) if dur else None,
    }


def save_upload_to_tmp(upload) -> Path:
    """Persist Streamlit UploadedFile to a temp file and return its path."""
    suffix = Path(upload.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.getbuffer())
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def format_duration(seconds: float | None) -> str:
    if not seconds:
        return "—"
    m = int(seconds // 60)
    s = int(round(seconds % 60))
    return f"{m:02d}:{s:02d}"