# app.py  (or streamlit_app.py)

"""
Minimal Streamlit UI for Traffic Sign & Speed Violation Detector.
- Sidebar: speed slider, sample picker, video upload
- Main: preview + meta, single 'Process Video' button
- Progress: shows decoding/inference progress (approx via frames)
- Results: annotated video player, small KPI chips, download buttons

Run:
    streamlit run app.py
"""

from __future__ import annotations
import os, time, cv2, streamlit as st, atexit, tempfile, io, pandas as pd
from pathlib import Path
from PIL import Image
from io import StringIO
from huggingface_hub import hf_hub_download

# ──────────────────────────────────────────────────────────────────────────────
# Page config MUST be the first Streamlit command
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Sign & Speed Violation Detector",
    page_icon="🚦",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# Ensure local imports (e.g., utils) resolve even if launched from elsewhere
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

# Local imports (expected in utils.py)
from utils import (
    process_video,
    get_sample_videos,
    get_video_meta,
    save_upload_to_tmp,
    format_duration,
)

# ──────────────────────────────────────────────────────────────────────────────
# Model weights (download from HF once; path returned is local)
# ──────────────────────────────────────────────────────────────────────────────
model_path = hf_hub_download(
    repo_id="WahburRehman/traffic-sign-detector",
    filename="model.pt"
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
ASSETS_DIR = ROOT / "assets"
SAMPLES_DIR = ASSETS_DIR / "samples"

# ──────────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────────
if "processing" not in st.session_state:
    st.session_state.processing = False
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "chosen_sample_path" not in st.session_state:
    st.session_state.chosen_sample_path = None
if "result_ver" not in st.session_state:
    st.session_state.result_ver = 0

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def video_thumb(path: Path) -> Image.Image | None:
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    scale = 200 / max(h, w) if max(h, w) > 200 else 1.0
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return Image.fromarray(rgb)


def render_video_meta_compact(meta: dict | None):
    if not meta:
        st.write("-")
        return
    fps = f"{meta.get('fps', '—')}"
    frames = f"{meta.get('frames', '—')}"
    res = meta.get('resolution', '—')
    dur_s = meta.get('duration_sec')
    dur = format_duration(dur_s)

    st.markdown("""
    <style>
    .kpi-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
    .kpi{
      padding:10px 12px;border:1px solid #EEE;border-radius:12px;
      background:rgba(0,0,0,0.02);
    }
    .kpi .label{font-size:12px;color:#666;margin-bottom:4px;}
    .kpi .val{font-size:18px;font-weight:600;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi"><div class="label">FPS</div><div class="val">{fps}</div></div>
      <div class="kpi"><div class="label">Frames</div><div class="val">{frames}</div></div>
      <div class="kpi"><div class="label">Resolution</div><div class="val">{res}</div></div>
      <div class="kpi"><div class="label">Duration</div><div class="val">{dur}</div></div>
    </div>
    """, unsafe_allow_html=True)


def render_results():
    res = st.session_state.get("result")
    if not res:
        return

    # KPIs
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Total detections", int(res["kpis"].get("total_detections", 0)))
    with c2: st.metric("Total violations", int(res["kpis"].get("total_violations", 0)))
    with c3: st.metric("Top class", str(res["kpis"].get("top_class") or "—"))

    # Always refresh preview temp file for the current result
    old = st.session_state.pop("preview_temp_path", None)
    if old and os.path.exists(old):
        try:
            os.unlink(old)
        except:
            pass

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(res["video_bytes"])
        st.session_state.preview_temp_path = f.name

    # Annotated video (fresh temp path each time)
    st.subheader("Annotated output")
    st.video(st.session_state.preview_temp_path)

    # Downloads
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "Download annotated MP4",
            data=res["video_bytes"],
            file_name=f"traffic-result-{time.strftime('%Y%m%d-%H%M%S')}.mp4",
            mime="video/mp4",
            width='stretch',
            key="dl_mp4",
        )
    with col_b:
        st.download_button(
            "Download events CSV",
            data=res["csv_bytes"],
            file_name=f"violations-{time.strftime('%Y%m%d-%H%M%S')}.csv",
            mime="text/csv",
            width='stretch',
            key="dl_csv",
        )

    # Optional CSV preview
    with st.expander("Violations timeline (CSV preview)", expanded=False):
        try:
            df = pd.read_csv(io.BytesIO(res["csv_bytes"]))
            st.dataframe(df.head(50), width='stretch', hide_index=True)
        except Exception:
            st.write("Could not preview CSV.")


def cleanup_temp_files():
    if "preview_temp_path" in st.session_state:
        try:
            if os.path.exists(st.session_state.preview_temp_path):
                os.unlink(st.session_state.preview_temp_path)
        except:
            pass

atexit.register(cleanup_temp_files)

# ──────────────────────────────────────────────────────────────────────────────
# Minimal styling
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
div.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Page header
# ──────────────────────────────────────────────────────────────────────────────
st.title("🚦 Traffic Sign & Speed Violation Detector")
st.caption("Select or upload a video, set speed, and process to get an annotated output with detections.")

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    speed_kmh = st.slider("Vehicle speed (km/h)", 0, 200, 50, 1)

    st.subheader("Sample videos")
    samples = get_sample_videos(SAMPLES_DIR)
    uploaded = None

    if samples:
        cols = st.columns(1) if len(samples) == 1 else st.columns(2)
        for i, p in enumerate(samples):
            col = cols[i % len(cols)]
            with col:
                with st.container():
                    img = video_thumb(p)
                    if img is not None:
                        st.image(img, caption=p.name, width='stretch')

                    if st.button("Select", key=f"use_{p.stem}", width='stretch'):
                        st.session_state.chosen_sample_path = str(p)
                        st.session_state.uploaded_file = None
                        st.session_state.pop("result", None)
                        st.session_state.pop("preview_temp_path", None)
                        st.session_state.processing = False
                        st.rerun()
    else:
        st.caption("No samples found in ./assets/samples")

    st.divider()

    uploaded = st.file_uploader(" ", type=["mp4", "mov", "avi", "mkv"], key="file_uploader")
    st.markdown("**Format tip:** MP4 (H.264) at 720p works best. Size ≤ 50 MB.")
    if uploaded is not None:
        # Only react if a *new* file arrived
        if st.session_state.uploaded_file != uploaded:
            # RESET STATE COMPLETELY when uploading a new file
            st.session_state.uploaded_file = uploaded
            st.session_state.chosen_sample_path = None
            st.session_state.pop("result", None)
            st.session_state.pop("preview_temp_path", None)
            st.session_state.processing = False
            st.rerun()
    else:
        # Also reset when file uploader is cleared
        if st.session_state.uploaded_file is not None:
            st.session_state.uploaded_file = None
            st.session_state.pop("result", None)
            st.session_state.pop("preview_temp_path", None)
            st.session_state.processing = False
            st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Decide input path (upload has precedence over sample)
# ──────────────────────────────────────────────────────────────────────────────
input_path: Path | None = None

if st.session_state.get("uploaded_file") is not None:
    input_path = save_upload_to_tmp(st.session_state.uploaded_file)
elif st.session_state.get("chosen_sample_path"):
    input_path = Path(st.session_state.chosen_sample_path)

# ──────────────────────────────────────────────────────────────────────────────
# Main layout: preview + metadata
# ──────────────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([2, 1], gap="large")

with col_left:
    st.subheader("Preview")
    if input_path is not None and input_path.exists():
        # If stale caching ever appears, switch to: st.video(Path(input_path).read_bytes())
        st.video(str(input_path))
    else:
        st.info("Pick a sample or upload a video to begin.")

with col_right:
    st.subheader("Video info")
    if input_path is not None and input_path.exists():
        meta = get_video_meta(input_path)
        render_video_meta_compact(meta)
    else:
        st.write("-")

# ──────────────────────────────────────────────────────────────────────────────
# Process button
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
process_disabled = (input_path is None) or st.session_state.processing
process_clicked = st.button("▶️ Process Video", type="primary", width='stretch', disabled=process_disabled)

if process_clicked and not st.session_state.processing:
    st.session_state.processing = True
    st.session_state.pop("result", None)
    old = st.session_state.pop("preview_temp_path", None)
    if old and os.path.exists(old):
        try:
            os.unlink(old)
        except:
            pass
    st.session_state.processing = True
    st.rerun()

# Placeholders for progress
prog = st.empty()
prog_text = st.empty()

# ──────────────────────────────────────────────────────────────────────────────
# Run pipeline when processing is flagged
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state.processing and input_path is not None:
    st.session_state.processing = True
    st.session_state.pop("result", None)

    prog.empty()
    prog_text.empty()

    if not os.path.exists(model_path):
        st.error(f"Model weights not found: {model_path}")
        st.session_state.processing = False
        st.stop()

    progress_bar = prog.progress(0, text="Starting…")
    start_time = time.time()

    def progress_cb(done: int, total: int, fps_est: float):
        pct = int((done / total) * 100) if total and total > 0 else 0
        progress_bar.progress(min(max(pct, 0), 100), text=f"Processing… {pct}%")
        prog_text.caption(f"Frames: {done}/{total}  |  Processing speed: {fps_est:.1f} FPS")

    try:
        # Process to a temporary file; we'll read bytes and discard the file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as temp_output:
            kpis, events = process_video(
                video_path=str(input_path),
                model_path=str(model_path),
                user_speed_kmh=float(speed_kmh),
                output_path=temp_output.name,
                progress_cb=progress_cb,
            )
            temp_output.seek(0)
            video_bytes = temp_output.read()

        # CSV bytes
        df = pd.DataFrame(events)
        csv_buf = StringIO()
        df.to_csv(csv_buf, index=False)
        csv_bytes = csv_buf.getvalue().encode("utf-8")

    except Exception as e:
        prog.empty()
        prog_text.empty()
        st.session_state.processing = False
        st.exception(e)
        st.stop()

    # Save result to state
    st.session_state["result"] = {
        "video_bytes": video_bytes,
        "csv_bytes": csv_bytes,
        "kpis": kpis,
    }
    st.session_state.result_ver += 1

    total_secs = time.time() - start_time
    progress_bar.progress(100, text="Done")
    prog_text.caption(f"Completed in {total_secs:.1f}s")
    st.session_state.processing = False

# ──────────────────────────────────────────────────────────────────────────────
# Render results if available
# ──────────────────────────────────────────────────────────────────────────────
if "result" in st.session_state and not st.session_state.processing:
    render_results()
