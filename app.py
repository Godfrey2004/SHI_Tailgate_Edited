import os
# Force OpenCV FFmpeg to low-latency, zero-buffer mode for RTSP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|"
    "fflags;nobuffer|"
    "flags;low_delay|"
    "max_delay;0|"
    "reorder_queue_size;0|"
    "probesize;32|"
    "analyzeduration;0"
)

import re, cv2, time, ctypes, threading, sys, atexit, logging, subprocess
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename


# ── Logging ──────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/app.log"),
    ],
)
log = logging.getLogger("SHI")
logging.getLogger("werkzeug").setLevel(logging.ERROR)

from datetime import datetime
import shutil
try:
    from generate_report import create_inspection_report
except ImportError:
    log.warning("generate_report.py not found. PDF generation disabled.")


# ── IKapC SDK (I-TEK industrial camera) ──────────────────────────────────────
IKAPC_DIR = r"C:\Program Files\I-TEK OptoElectronics\IKapLibrary\Examples\Python\IKapLib"
if IKAPC_DIR not in sys.path:
    sys.path.append(IKAPC_DIR)
try:
    import IKapC, IKapCDef
    SDK_AVAILABLE = True
    log.info("IKapC SDK loaded OK")
except ImportError:
    SDK_AVAILABLE = False
    log.warning("IKapC SDK NOT found - industrial camera disabled")

# ── Flask app ─────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR    = os.path.join(BASE_DIR, "uploads")
DATA_DIR_NAME = "tailgate_data"
DATA_DIR      = os.path.join(BASE_DIR, DATA_DIR_NAME)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

import yaml

app = Flask(__name__, static_folder=BASE_DIR)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
_last_config_mtime = 0
_cached_config = {}

def load_config():
    global _last_config_mtime, _cached_config
    if os.path.exists(CONFIG_PATH):
        try:
            mtime = os.path.getmtime(CONFIG_PATH)
            if mtime != _last_config_mtime:
                with open(CONFIG_PATH, "r") as f:
                    _cached_config = yaml.safe_load(f) or {}
                _last_config_mtime = mtime
        except Exception as e:
            log.error(f"Error loading config.yaml: {e}")

def get_conf_threshold(cls_name):
    load_config()
    thresholds = _cached_config.get("confidence_thresholds", {})
    return thresholds.get(cls_name, thresholds.get("default", 0.5))

def get_timing(key, default):
    load_config()
    timings = _cached_config.get("timings", {})
    return float(timings.get(key, default))

def get_banner(key, default, **kwargs):
    load_config()
    banners = _cached_config.get("banners", {})
    template = banners.get(key, default)
    try:
        return template.format(**kwargs)
    except Exception as e:
        return default

lock = threading.RLock()
cycle_count = 1
zt = None
ocr_reset_flag = False

# ── Shared state ──────────────────────────────────────────────────────────────
current_cycle = {
    "status"          : "Awaiting camera connection",
    "is_processing"   : False,
    "cycle_number"    : "#001",
    "serial"          : "------",
    "serial_date"     : "------",
    "serial_shift"    : "-",
    "serial_count"    : "---",
    "serial_time"     : "--:--",
    "confidence"      : "- -",
    "serial_finalized": False,  # True when OCR voting has locked the serial
    "ocr_phase"       : "scanning",
    "ocr_scan_start_time": time.time(),
    "ocr_fail_time"   : 0.0,
    "result"          : "Awaiting analysis...",
    "holes_count"     : 0,
    "defects"         : [],
    "step1_status"    : "Pending",
    "step2_status"    : "Pending",
    "step3_status"    : "Pending",
    "instruction"     : "WAITING FOR PART",
    "instruction_color": "blue",
    "zone_inner1_status"  : "waiting_for_zone",
    "zone_inner1_progress": 0,
    "zone_inner2_status"  : "waiting_for_zone",
    "zone_inner2_progress": 0,
    "zone_outer1_status"  : "waiting_for_zone",
    "zone_outer1_progress": 0,
    "zone_outer2_status"  : "waiting_for_zone",
    "zone_outer2_progress": 0,
    # ── Cycle Statistics ────────────────────────────────────────────────────
    # cycle_result: "idle" | "running" | "PASS" | "FAIL"
    "cycle_result"     : "idle",
    "total_cycles"     : 0,
    "pass_cycles"      : 0,
    "fail_cycles"      : 0,
    "traceability_done": False,
    "zone_images"      : {}, # Stores the path or frame for each zone
    # ── Empty Table Detection ───────────────────────────────────
    # Triggers new cycle ONLY when YOLO detects the 'Empty_table' class
    # continuously for 3 seconds during a running cycle.
    "empty_table_first_seen": None,  # time.time() when Empty_table class first detected
    "auto_finishing"         : False, # True while auto-finish is in progress
}

# ── CP Plus (RTSP) stream ─────────────────────────────────────────────────────
cp_cap              = None          # cv2.VideoCapture for CP Plus
cp_frame_lock       = threading.Lock()
cp_frame_event      = threading.Event()
cp_jpeg_cond        = threading.Condition()
cp_frame_seq        = 0
cp_latest_jpeg      = None          # latest JPEG bytes from CP Plus (1920x1080)
cp_latest_raw_frame = None          # latest raw cv2 frame from CP Plus (1920x1080)
cp_latest_frame     = None          # latest raw cv2 frame for detection_loop (1920x1080)
cp_stream_active    = False
latest_yolo_segs    = []            # list of {name, conf, xyxy, pts}

def cp_capture_loop():
    """Ultra-fast background thread: captures CP Plus RTSP at line rate with zero latency."""
    global cp_cap, cp_latest_raw_frame, cp_latest_frame, cp_stream_active
    consecutive_fail = 0
    log.info("[CP] Dedicated RTSP capture loop started")
    while True:
        start_t = time.time()
        with cp_frame_lock:
            cap = cp_cap
            active = cp_stream_active
        if not active or cap is None:
            time.sleep(0.05)
            continue

        ret, frame = cap.read()
        if not ret:
            consecutive_fail += 1
            if consecutive_fail > 60:
                log.warning("[CP] Too many read failures - marking stream inactive")
                with cp_frame_lock:
                    cp_stream_active = False
                consecutive_fail = 0
            time.sleep(0.02)
            continue
        consecutive_fail = 0

        # Guarantee 1920x1080 Full HD resolution for feed and processing
        h, w = frame.shape[:2]
        if w != 1920 or h != 1080:
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)

        with cp_frame_lock:
            cp_latest_raw_frame = frame
            cp_latest_frame = frame
        cp_frame_event.set()
        
        elapsed = time.time() - start_t
        target_fps = 1.0 / 30.0
        if elapsed < target_fps:
            time.sleep(target_fps - elapsed)

threading.Thread(target=cp_capture_loop, daemon=True).start()

def cp_display_loop():
    """Background thread: renders overlays and encodes 1920x1080 JPEG at smooth ~30 fps."""
    global cp_latest_jpeg, cp_frame_seq
    log.info("[CP] Display encoding loop started")
    last_processed_time = 0.0
    TARGET_INTERVAL = 1.0 / 30.0  # Smooth 30 FPS target

    # Colour palette per class (BGR) — matched to new model class names
    CLASS_COLORS = {
        "inner_zone_1"     : (255, 200,   0),   # Blue
        "inner_zone_2"     : (0,   255,   0),   # Green
        "outer_zone_1"     : (255,  50, 255),   # Magenta/Pink
        "outer_zone_2"     : (0,   165, 255),   # Orange
        "hand"             : (40,   40, 255),   # Red
        "person"           : (0,   100, 255),   # Orange-Red
        "empty_table"      : (120, 120, 120),   # Grey
        "part_onthe_table" : (0,   220, 100),   # Lime Green
    }
    DEFAULT_COLOR = (180, 230, 180)

    while True:
        # Wait for a new frame event with timeout
        cp_frame_event.wait(timeout=0.04)
        cp_frame_event.clear()

        now = time.time()
        elapsed = now - last_processed_time
        if elapsed < TARGET_INTERVAL:
            time.sleep(max(0.001, TARGET_INTERVAL - elapsed))

        with cp_frame_lock:
            frame = cp_latest_raw_frame
            active = cp_stream_active
            segs_to_draw = list(latest_yolo_segs)

        if not active or frame is None:
            time.sleep(0.05)
            continue

        last_processed_time = time.time()

        if segs_to_draw:
            draw_frame = frame.copy()
            for seg in segs_to_draw:
                cls_name = seg["name"]
                # Normalise to lowercase+underscores to match CLASS_COLORS keys
                color    = CLASS_COLORS.get(cls_name.lower().replace(" ", "_"), DEFAULT_COLOR)
                
                # Draw thick bounding box instead of mask
                x1, y1, x2, y2 = seg["xyxy"]
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 4)
                lx, ly = x1, max(y1 - 10, 10)

                # Label without confidence threshold
                label = f"{cls_name}"
                font_scale = 1.0
                thickness = 2
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Label background
                cv2.rectangle(draw_frame, (lx, ly - th - 8), (lx + tw + 12, ly + baseline + 4), color, -1)
                
                # Label text
                cv2.putText(draw_frame, label, (lx + 6, ly), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        else:
            draw_frame = frame

        # (Banner drawing removed to avoid double-banner with HTML UI)
        
        # JPEG encode at quality 70 (crisp 1080p, low network payload)
        _, buf = cv2.imencode(".jpg", draw_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        jpeg_bytes = buf.tobytes()

        with cp_jpeg_cond:
            cp_latest_jpeg = jpeg_bytes
            cp_frame_seq += 1
            cp_jpeg_cond.notify_all()

threading.Thread(target=cp_display_loop, daemon=True).start()

def gen_cp_frames():
    """MJPEG generator for CP Plus stream - yields only fresh frames with zero duplicate flooding."""
    placeholder = _make_placeholder("CP Plus - NO SIGNAL")
    last_seq = -1
    while True:
        with cp_jpeg_cond:
            if cp_frame_seq == last_seq or cp_latest_jpeg is None:
                cp_jpeg_cond.wait(timeout=0.2)
            jpeg = cp_latest_jpeg
            cur_seq = cp_frame_seq

        if jpeg is not None and cur_seq != last_seq:
            last_seq = cur_seq
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        elif jpeg is None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + placeholder + b"\r\n")
            time.sleep(0.2)

# ── Industrial camera (IKapC GigE) ───────────────────────────────────────────
class CameraStreamer:
    """I-TEK IKapC GigE camera wrapper (ported from reference app.py)."""

    def __init__(self):
        self.m_hDev            = ctypes.c_void_p(None)
        self.m_hStream         = ctypes.c_void_p(None)
        self.m_hBufferConvert  = ctypes.c_void_p(None)
        self.actual_pixel_format = ""
        self.cam_width         = 0
        self.cam_height        = 0
        self.latest_frame      = None
        self.EndOfFrameProc    = None
        self._lock             = threading.Lock()
        self._diag_count       = 0
        self.is_connected      = False
        self.last_error_msg    = ""
        if SDK_AVAILABLE:
            res = IKapC.ItkManInitialize()
            if res != IKapCDef.ITKSTATUS_OK:
                self.last_error_msg = f"SDK init failed: {res}"
                log.error(self.last_error_msg)

    # ── scan ─────────────────────────────────────────────────────────────────
    def scan_cameras(self):
        if not SDK_AVAILABLE:
            return []
        res, n = IKapC.ItkManGetDeviceCount()
        if res != IKapCDef.ITKSTATUS_OK or n == 0:
            return []
        cams = []
        for i in range(n):
            res, info = IKapC.ItkManGetDeviceInfo(i)
            if res == IKapCDef.ITKSTATUS_OK:
                cams.append({
                    "id"           : i,
                    "model"        : info.FullName.decode("utf-8", errors="ignore"),
                    "display_name" : info.UserDefinedName.decode("utf-8", errors="ignore") or f"Cam{i}",
                })
        return cams

    # ── connect ───────────────────────────────────────────────────────────────
    def connect(self, index=0, exposure=None, gain=None, gamma=None,
                pixel_format=None, trigger_mode=None):
        if not SDK_AVAILABLE:
            self.last_error_msg = "IKapC SDK unavailable"
            return False
        if self.is_connected:
            self.disconnect()

        res, self.m_hDev = IKapC.ItkDevOpen(index, IKapCDef.ITKDEV_VAL_ACCESS_MODE_EXCLUSIVE)
        if res != IKapCDef.ITKSTATUS_OK or not self.m_hDev or self.m_hDev.value == 0:
            self.last_error_msg = f"Cannot open device (err={res})"
            return False

        # Apply settings
        try:
            if exposure  : IKapC.ItkDevSetDouble(self.m_hDev, b"ExposureTime", float(exposure))
            if gain      : IKapC.ItkDevSetDouble(self.m_hDev, b"Gain",         float(gain))
            if gamma     : IKapC.ItkDevSetDouble(self.m_hDev, b"Gamma",        float(gamma))
        except Exception as e:
            log.warning(f"[GigE] Setting apply warning: {e}")

        pf = pixel_format or "BayerRG8"
        try:
            IKapC.ItkDevFromString(self.m_hDev, b"PixelFormat", pf.encode())
        except Exception:
            pass
        if trigger_mode:
            try:
                IKapC.ItkDevFromString(self.m_hDev, b"TriggerMode", trigger_mode.encode())
            except Exception:
                pass

        # Read back actual pixel format and dimensions
        try:
            _, pf_bytes = IKapC.ItkDevToString(self.m_hDev, b"PixelFormat")
            self.actual_pixel_format = pf_bytes.decode("utf-8", errors="ignore") if isinstance(pf_bytes, bytes) else str(pf_bytes)
        except Exception:
            self.actual_pixel_format = pf

        for attr, key in [("cam_width", b"Width"), ("cam_height", b"Height")]:
            try:
                _, v = IKapC.ItkDevToString(self.m_hDev, key)
                setattr(self, attr, int(v.decode() if isinstance(v, bytes) else str(v)))
            except Exception:
                pass

        # Allocate stream
        res, self.m_hStream = IKapC.ItkDevAllocStreamEx(self.m_hDev, 0, 5)
        if res != IKapCDef.ITKSTATUS_OK:
            self.last_error_msg = "Failed to allocate stream"
            return False

        # Bayer conversion buffer
        bayer_fmts = ["BayerRG8","BayerBG8","BayerGR8","BayerGB8"]
        if self.actual_pixel_format in bayer_fmts and self.cam_width > 0:
            res, self.m_hBufferConvert = IKapC.ItkBufferNew(
                self.cam_width, self.cam_height, IKapCDef.ITKBUFFER_VAL_FORMAT_BGR888)
        else:
            self.m_hBufferConvert = ctypes.c_void_p(None)

        # Stream mode
        IKapC.ItkStreamSetPrm(self.m_hStream, IKapCDef.ITKSTREAM_PRM_TRANSFER_MODE,
                               ctypes.c_uint32(IKapCDef.ITKSTREAM_VAL_TRANSFER_MODE_SYNCHRONOUS_WITH_PROTECT))
        IKapC.ItkStreamSetPrm(self.m_hStream, IKapCDef.ITKSTREAM_PRM_START_MODE,
                               ctypes.c_uint32(IKapCDef.ITKSTREAM_VAL_START_MODE_NON_BLOCK))

        # Register callback
        self.EndOfFrameProc = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(self._on_frame)
        IKapC.ItkStreamRegisterCallback(
            self.m_hStream, IKapCDef.ITKSTREAM_VAL_EVENT_TYPE_END_OF_FRAME,
            self.EndOfFrameProc, ctypes.c_void_p(None))

        res = IKapC.ItkStreamStart(self.m_hStream, 0)
        if res == IKapCDef.ITKSTATUS_OK:
            self.is_connected   = True
            self.last_error_msg = ""
            log.info("[GigE] Camera connected OK")
            return True
        self.last_error_msg = "Failed to start stream"
        return False

    # ── frame callback ────────────────────────────────────────────────────────
    def _on_frame(self, pParam):
        res, hBuffer = IKapC.ItkStreamGetCurrentBuffer(self.m_hStream)
        if res != IKapCDef.ITKSTATUS_OK:
            return
        res, info = IKapC.ItkBufferGetInfo(hBuffer)
        if info.State not in (IKapCDef.ITKBUFFER_VAL_STATE_FULL,
                               IKapCDef.ITKBUFFER_VAL_STATE_UNCOMPLETED):
            return

        w, h = info.ImageWidth, info.ImageHeight
        pf   = self.actual_pixel_format.upper()

        if pf.startswith("BAYER"):
            # Try SDK convert first
            if self.m_hBufferConvert and self.m_hBufferConvert.value:
                cr = IKapC.ItkBufferConvert(hBuffer, self.m_hBufferConvert,
                                             IKapCDef.ITKBUFFER_VAL_FORMAT_BGR888,
                                             IKapCDef.ITKBUFFER_VAL_CONVERT_OPTION_AUTO_FORMAT)
                if cr == IKapCDef.ITKSTATUS_OK:
                    r2, np_arr = IKapC.ItkBufferToNumPy(self.m_hBufferConvert)
                    if r2 == IKapCDef.ITKSTATUS_OK and np_arr is not None:
                        if len(np_arr.shape) == 1:
                            np_arr = np_arr.reshape((h, w, 3))
                        with self._lock:
                            self.latest_frame = np_arr.copy()
                        return
            # OpenCV Bayer fallback
            r3, raw = IKapC.ItkBufferToNumPy(hBuffer)
            if r3 == IKapCDef.ITKSTATUS_OK and raw is not None:
                if len(raw.shape) == 1:
                    raw = raw.reshape((h, w))
                bayer_map = {"BAYERRG8": cv2.COLOR_BayerRG2BGR,
                             "BAYERBG8": cv2.COLOR_BayerBG2BGR,
                             "BAYERGR8": cv2.COLOR_BayerGR2BGR,
                             "BAYERGB8": cv2.COLOR_BayerGB2BGR}
                code = bayer_map.get(pf, cv2.COLOR_BayerRG2BGR)
                with self._lock:
                    self.latest_frame = cv2.cvtColor(raw, code)

        elif pf == "MONO8":
            r3, raw = IKapC.ItkBufferToNumPy(hBuffer)
            if r3 == IKapCDef.ITKSTATUS_OK and raw is not None:
                if len(raw.shape) == 1:
                    raw = raw.reshape((h, w))
                with self._lock:
                    self.latest_frame = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

        elif pf in ("BGR8", "BGR24", "RGB8"):
            r3, raw = IKapC.ItkBufferToNumPy(hBuffer)
            if r3 == IKapCDef.ITKSTATUS_OK and raw is not None:
                if len(raw.shape) == 1:
                    raw = raw.reshape((h, w, 3))
                frame = raw if pf != "RGB8" else cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
                with self._lock:
                    self.latest_frame = frame.copy()

    def get_frame(self):
        with self._lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    # ── update features ───────────────────────────────────────────────────────
    def update_features(self, exposure=None, gain=None, gamma=None,
                        trigger_mode=None, **kw):
        if not self.is_connected or not self.m_hDev or self.m_hDev.value == 0:
            return False, "Camera not connected"
        try:
            if exposure    : IKapC.ItkDevSetDouble(self.m_hDev, b"ExposureTime", float(exposure))
            if gain        : IKapC.ItkDevSetDouble(self.m_hDev, b"Gain",         float(gain))
            if gamma       : IKapC.ItkDevSetDouble(self.m_hDev, b"Gamma",        float(gamma))
            if trigger_mode: IKapC.ItkDevFromString(self.m_hDev, b"TriggerMode", trigger_mode.encode())
        except Exception as e:
            return False, str(e)
        return True, "Features updated"

    # ── disconnect ────────────────────────────────────────────────────────────
    def disconnect(self):
        if not SDK_AVAILABLE:
            return
        if self.m_hStream and self.m_hStream.value:
            try:
                IKapC.ItkStreamStop(self.m_hStream)
                IKapC.ItkStreamUnregisterCallback(
                    self.m_hStream, IKapCDef.ITKSTREAM_VAL_EVENT_TYPE_END_OF_FRAME)
                IKapC.ItkDevFreeStream(self.m_hStream)
            except Exception as e:
                log.error(f"[GigE] Stream free error: {e}")
            self.m_hStream = ctypes.c_void_p(None)
        if self.m_hDev and self.m_hDev.value:
            try:
                IKapC.ItkDevClose(self.m_hDev)
            except Exception as e:
                log.error(f"[GigE] Device close error: {e}")
            self.m_hDev = ctypes.c_void_p(None)
        if self.m_hBufferConvert and self.m_hBufferConvert.value:
            try:
                IKapC.ItkBufferFree(self.m_hBufferConvert)
            except Exception:
                pass
            self.m_hBufferConvert = ctypes.c_void_p(None)
        self.is_connected   = False
        self.latest_frame   = None
        log.info("[GigE] Disconnected")


cam = CameraStreamer()
atexit.register(cam.disconnect)

# Auto-start standalone OCR Engine
ocr_process = None
try:
    log.info("Starting standalone OCR process in background...")
    ocr_log_file = open(os.path.join(BASE_DIR, "logs", "ocr.log"), "a")
    ocr_process = subprocess.Popen(
        [sys.executable, "ocr_processor.py"],
        stdout=ocr_log_file,
        stderr=subprocess.STDOUT
    )
    
    def cleanup_ocr_process():
        if ocr_process:
            log.info("Terminating standalone OCR process...")
            ocr_process.terminate()
            try:
                ocr_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                ocr_process.kill()
            
    atexit.register(cleanup_ocr_process)
except Exception as e:
    log.error(f"Failed to start standalone OCR process: {e}")

current_ocr_box = None  # Receives coordinates from the standalone OCR process
ind_latest_raw_frame = None  # Stored globally for the standalone OCR process to fetch

# ── Industrial feed MJPEG generator ──────────────────────────────────────────
ind_latest_jpeg   = None
ind_jpeg_lock     = threading.Lock()
ind_jpeg_cond     = threading.Condition()
ind_frame_seq     = 0
ind_video_cap     = None
ind_video_active  = False

def ind_capture_loop():
    global ind_latest_jpeg, ind_video_cap, ind_video_active, ind_frame_seq, ind_latest_raw_frame
    log.info("[IND] Capture loop started")
    consecutive_fail = 0
    while True:
        start_t = time.time()
        frame = None
        if ind_video_active and ind_video_cap is not None:
            ret, frame = ind_video_cap.read()
            if not ret:
                consecutive_fail += 1
                if consecutive_fail > 60:
                    ind_video_active = False
                    log.warning("[IND] Video ended")
                    time.sleep(1.0)
                continue
            else:
                consecutive_fail = 0
        else:
            if not cam.is_connected:
                time.sleep(0.05)
                continue
            frame = cam.get_frame()
            
        if frame is not None:
            ind_latest_raw_frame = frame.copy()
            if current_ocr_box:
                x1, y1, x2, y2 = current_ocr_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "OCR CROP", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with ind_jpeg_cond:
                ind_latest_jpeg = buf.tobytes()
                ind_frame_seq += 1
                ind_jpeg_cond.notify_all()
                
        elapsed = time.time() - start_t
        target_fps = 1.0 / 30.0
        if elapsed < target_fps:
            time.sleep(target_fps - elapsed)

threading.Thread(target=ind_capture_loop, daemon=True).start()

def gen_ind_frames():
    placeholder = _make_placeholder("Industrial - NO SIGNAL")
    last_seq = -1
    while True:
        with ind_jpeg_cond:
            if ind_frame_seq == last_seq or ind_latest_jpeg is None:
                ind_jpeg_cond.wait(timeout=0.2)
            jpeg = ind_latest_jpeg
            cur_seq = ind_frame_seq
        if jpeg is not None and cur_seq != last_seq:
            last_seq = cur_seq
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + jpeg + b"\r\n")
        elif jpeg is None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + placeholder + b"\r\n")
            time.sleep(0.2)

# ── Placeholder helper ────────────────────────────────────────────────────────
_placeholder_cache = {}
def _make_placeholder(text="NO SIGNAL"):
    if text not in _placeholder_cache:
        img = np.full((480, 640, 3), 30, dtype=np.uint8)
        cv2.putText(img, text, (60, 250), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (100, 100, 100), 2, cv2.LINE_AA)
        _, buf = cv2.imencode(".jpg", img)
        _placeholder_cache[text] = buf.tobytes()
    return _placeholder_cache[text]

# ── Flask Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

# ── CP Plus stream ────────────────────────────────────────────────────────────
@app.route("/video_feed")
@app.route("/video_feed_cp")
def video_feed():
    return Response(gen_cp_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/connect_camera", methods=["POST"])
def connect_camera():
    """Connect CP Plus camera via RTSP credentials."""
    global cp_cap, cp_stream_active
    data = request.get_json(silent=True) or {}
    ip   = (data.get("ip") or "").strip() or "192.10.66.160"
    port = data.get("port", 554)
    user = data.get("user", "")
    pwd  = data.get("pwd",  "")
    path = data.get("path", "/stream1")

    rtsp_url = f"rtsp://{user}:{pwd}@{ip}:{port}{path}"
    rtsps_url = f"rtsps://{user}:{pwd}@{ip}:{port}{path}"
    log.info(f"[CP] Connecting to rtsp(s)://{user}:***@{ip}:{port}{path}")

    with cp_frame_lock:
        cp_stream_active = False
        if cp_cap:
            cp_cap.release()
        cp_cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cp_cap.isOpened():
            log.info("[CP] rtsp connection failed, trying rtsps (secure RTSP)...")
            cp_cap = cv2.VideoCapture(rtsps_url, cv2.CAP_FFMPEG)
        if not cp_cap.isOpened():
            log.error("[CP] Failed to open RTSP stream")
            return jsonify({"status": "error", "message": "Cannot open RTSP stream. Check credentials/IP."}), 500
        cp_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cp_cap.set(cv2.CAP_PROP_FPS, 30)
        cp_stream_active = True

    with lock:
        current_cycle["is_processing"] = True
        current_cycle["status"] = f"CP Plus connected ({ip})"

    log.info("[CP] RTSP stream connected OK")
    return jsonify({"status": "success", "message": "CP Plus camera connected"})

@app.route("/disconnect_camera", methods=["POST"])
def disconnect_camera():
    global cp_cap, cp_stream_active
    with cp_frame_lock:
        cp_stream_active = False
        if cp_cap:
            cp_cap.release()
            cp_cap = None
    with lock:
        current_cycle["is_processing"] = False
        current_cycle["status"] = "Awaiting camera connection"
    return jsonify({"status": "success", "message": "CP Plus disconnected"})

# ── Video file upload ─────────────────────────────────────────────────────────
@app.route("/upload_video", methods=["POST"])
def upload_video():
    global cp_cap, cp_stream_active
    if "video" not in request.files:
        return jsonify({"status": "error", "message": "No file provided"}), 400
    f        = request.files["video"]
    filename = secure_filename(f.filename)
    if not filename:
        return jsonify({"status": "error", "message": "Invalid filename"}), 400
    path = os.path.join(UPLOAD_DIR, filename)
    f.save(path)
    with cp_frame_lock:
        cp_stream_active = False
        if cp_cap:
            cp_cap.release()
        cp_cap = cv2.VideoCapture(path)
        if not cp_cap.isOpened():
            return jsonify({"status": "error", "message": "Cannot open video"}), 500
        cp_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cp_stream_active = True
    with lock:
        current_cycle["is_processing"] = True
        current_cycle["status"] = f"Streaming: {filename}"
    return jsonify({"status": "success", "message": f"Streaming {filename}"})

@app.route("/upload_ind_video", methods=["POST"])
def upload_ind_video():
    global ind_video_cap, ind_video_active
    if "video" not in request.files:
        return jsonify({"status": "error", "message": "No file provided"}), 400
    f        = request.files["video"]
    filename = secure_filename(f.filename)
    if not filename:
        return jsonify({"status": "error", "message": "Invalid filename"}), 400
    path = os.path.join(UPLOAD_DIR, filename)
    f.save(path)
    
    ind_video_active = False
    if ind_video_cap:
        ind_video_cap.release()
    ind_video_cap = cv2.VideoCapture(path)
    if not ind_video_cap.isOpened():
        return jsonify({"status": "error", "message": "Cannot open video"}), 500
    ind_video_active = True
    
    return jsonify({"status": "success", "message": "Industrial Video loaded successfully"})

# ── Industrial / GigE routes ──────────────────────────────────────────────────
@app.route("/industrial_feed")
def industrial_feed():
    return Response(gen_ind_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/scan_cameras")
def scan_cameras():
    return jsonify({"cameras": cam.scan_cameras()})

@app.route("/connect_gige", methods=["POST"])
def connect_gige():
    if cam.is_connected:
        return jsonify({"status": "success", "message": "Already connected"})
    data = request.get_json(silent=True) or {}
    
    # Load default exposure from config if not provided
    exposure = data.get("exposure")
    if exposure is None:
        load_config()
        exposure = _cached_config.get("camera", {}).get("industrial", {}).get("exposure")

    target_index = int(data.get("index", 0))
    
    # ── ENFORCE SPECIFIC CAMERA ──
    available_cams = cam.scan_cameras()
    found_target = False
    
    # First, check if the UI's selected index is indeed the correct camera model
    for c in available_cams:
        if c["id"] == target_index and ("UA5MEAGV" in c["model"] or "UA5MEAGV" in c["display_name"]):
            found_target = True
            break
            
    # If the UI's index wasn't correct, auto-search for the correct camera
    if not found_target:
        for c in available_cams:
            if "UA5MEAGV" in c["model"] or "UA5MEAGV" in c["display_name"]:
                target_index = c["id"]
                found_target = True
                break
            
    if not found_target and len(available_cams) > 0:
        return jsonify({"status": "error", "message": "Camera UA5MEAGV-50M not found. Discovered other cameras but blocked connection to prevent mismatch."}), 400

    ok   = cam.connect(
        index        = target_index,
        exposure     = exposure,
        gain         = data.get("gain"),
        gamma        = data.get("gamma"),
        pixel_format = data.get("pixel_format"),
        trigger_mode = data.get("trigger_mode"),
    )
    if ok:
        return jsonify({"status": "success", "message": "Industrial camera connected"})
    return jsonify({"status": "error", "message": cam.last_error_msg}), 500

@app.route("/disconnect_gige", methods=["POST"])
def disconnect_gige():
    try:
        cam.disconnect()
        return jsonify({"status": "success", "message": "Industrial camera disconnected"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/update_gige_features", methods=["POST"])
def update_gige_features():
    data    = request.get_json(silent=True) or {}
    ok, msg = cam.update_features(**data)
    return jsonify({"status": "success" if ok else "error", "message": msg})

@app.route("/camera_status")
def camera_status():
    return jsonify({
        "connected"  : cam.is_connected,
        "status"     : "Connected" if cam.is_connected else "Disconnected",
        "model"      : "IKap GigE" if cam.is_connected else "",
        "fps"        : 30 if cam.is_connected else None,
        "resolution" : f"{cam.cam_width}x{cam.cam_height}" if cam.is_connected else None,
    })

@app.route("/camera_diag")
def camera_diag():
    frame = cam.get_frame()
    return jsonify({
        "connected"           : cam.is_connected,
        "actual_pixel_format" : cam.actual_pixel_format,
        "cam_width"           : cam.cam_width,
        "cam_height"          : cam.cam_height,
        "frame_shape"         : list(frame.shape) if frame is not None else None,
    })

# ── Cycle logic helpers ───────────────────────────────────────────────────────
ALL_ZONES = ("inner1", "inner2", "outer1", "outer2")

# ── Zone Configuration ────────────────────────────────────────────────────────
# inspect_time : Exactly 3.0 seconds accumulated time for all zones.
ZONE_CONFIG = {
    "inner1": {"name": "Inner Zone 1", "inspect_time": 3.0},
    "inner2": {"name": "Inner Zone 2", "inspect_time": 3.0},
    "outer1": {"name": "Outer Zone 1", "inspect_time": 3.0},
    "outer2": {"name": "Outer Zone 2", "inspect_time": 3.0},
}

# ── YOLO Zone Detection Worker ────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    zone_model_path = os.path.join(BASE_DIR, "Models", "sequence", "SHI_SEQ_V3.pt")
    log.info(f"Loading Zone YOLO model ({zone_model_path})...")
    zone_model = YOLO(zone_model_path)
except Exception as e:
    log.error(f"Failed to load Zone YOLO model: {e}")
    zone_model = None

# Maps internal zone keys → YOLO model class names (lowercase+underscore normalised)
YOLO_CLASS_MAP = {
    "inner1": "inner_zone_1",
    "inner2": "inner_zone_2",
    "outer1": "outer_zone_1",
    "outer2": "outer_zone_2",
}



def _get_next_pending_zone():
    """Return the first zone that is not 'done', or None if all done."""
    for z in ALL_ZONES:
        if current_cycle[f"zone_{z}_status"] != "done":
            return z
    return None


def _get_or_create_cycle_dir():
    """Returns (final_dir, folder_name) for the active cycle, creating the folder on disk.
    
    Folder naming rules:
    - Serial captured : <serial_number>
    - No serial       : <HHmmss>_no_serial
    """
    if "final_dir" in current_cycle and current_cycle["final_dir"]:
        serial_str = current_cycle.get("serial", "")
        has_real_serial = serial_str not in ("------", "", None, "NO_SERIAL")
        current_folder = current_cycle.get("folder_name", "temp")
        
        if has_real_serial and current_folder.endswith("_no_serial"):
            old_dir = current_cycle["final_dir"]
            # Clean serial string for folder name
            safe_serial = "".join(c for c in serial_str if c.isalnum() or c in (' ', '-', '_')).strip()
            new_folder = safe_serial
            new_dir = os.path.join(os.path.dirname(old_dir), new_folder)
            
            if os.path.exists(new_dir):
                timestamp_str = datetime.now().strftime("%H%M%S")
                new_folder = f"{new_folder}_{timestamp_str}"
                new_dir = os.path.join(os.path.dirname(old_dir), new_folder)
                
            try:
                os.rename(old_dir, new_dir)
                for fname in os.listdir(new_dir):
                    if fname.startswith(current_folder):
                        new_fname = fname.replace(current_folder, new_folder, 1)
                        os.rename(os.path.join(new_dir, fname), os.path.join(new_dir, new_fname))
                
                current_cycle["final_dir"] = new_dir
                current_cycle["folder_name"] = new_folder
                log.info(f"[STORAGE] Renamed cycle directory to include serial: {new_dir}")
                return new_dir, new_folder
            except Exception as e:
                log.error(f"[STORAGE] Failed to rename directory: {e}")

        return current_cycle["final_dir"], current_folder
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    day_dir = os.path.join(DATA_DIR, today_str)
    os.makedirs(day_dir, exist_ok=True)
    
    serial_str = current_cycle.get("serial", "")
    if serial_str in ("------", "", None):
        # No serial: use timestamp + no_serial tag
        timestamp_str = datetime.now().strftime("%H%M%S")
        folder_name = f"{timestamp_str}_no_serial"
    else:
        folder_name = serial_str
    
    final_dir = os.path.join(day_dir, folder_name)
    if os.path.exists(final_dir) and not current_cycle.get("folder_created"):
        timestamp_str = datetime.now().strftime("%H%M%S")
        folder_name = f"{folder_name}_{timestamp_str}"
        final_dir = os.path.join(day_dir, folder_name)
        
    os.makedirs(final_dir, exist_ok=True)
    current_cycle["final_dir"] = final_dir
    current_cycle["folder_name"] = folder_name
    current_cycle["folder_created"] = True
    log.info(f"[STORAGE] Cycle directory ready: {final_dir}")
    return final_dir, folder_name


def _save_image_to_disk(name, img_data):
    """Saves an image (bytes or ndarray) immediately into the cycle directory."""
    if img_data is None:
        return None
    try:
        final_dir, folder_name = _get_or_create_cycle_dir()
        img_path = os.path.join(final_dir, f"{folder_name}_{name}.png")
        if isinstance(img_data, bytes):
            with open(img_path, "wb") as f:
                f.write(img_data)
        elif isinstance(img_data, np.ndarray):
            cv2.imwrite(img_path, img_data)
        log.info(f"[IMAGE] Immediately saved {name} → {img_path}")
        return img_path
    except Exception as e:
        log.error(f"[IMAGE] Failed to save {name} to disk: {e}")
        return None


def zone_inference_loop():
    """
    Production zone inference loop.

    Simplified zone state machine per zone:
      waiting_for_zone → capturing (hold for capture_delay with hand-check) → done (green)

    - capturing: Zone detected, countdown begins. If zone lost → back to waiting.
                 After capture_delay, hand check (except inner1). Photo saved → done.
    - done:      Photo captured, zone card turns green immediately.

    OCR (serial) runs fully in parallel on the industrial camera.
    Cycle finalizes when ALL zones done AND serial finalized (any order).
    """
    # Per-zone timing state
    global zt
    if 'zt' not in globals() or zt is None:
        zt = {z: {"detect_start": None, "last_seen": 0.0, "capture_start": None,
                  "accumulated": 0.0} for z in ALL_ZONES}

    while True:
        if zone_model is None or current_cycle.get("cycle_result") in ("PASS", "FAIL"):
            time.sleep(1.0)
            continue

        with cp_frame_lock:
            frame = cp_latest_frame

        if frame is None:
            time.sleep(0.1)
            continue

        try:
            # ── Run YOLO inference at imgsz=640 for fast CPU execution ────────
            results = zone_model(frame, verbose=False, imgsz=640)

            # Build segmentation draw list (mask polygons when available)
            detected_zone = None
            detected_zone_xyxy = None
            segs_to_save  = []
            if len(results) > 0 and len(results[0].boxes) > 0:
                best_conf = 0.0
                boxes = results[0].boxes
                masks = results[0].masks  # may be None for detection-only models
                for i in range(len(boxes)):
                    c_id     = int(boxes.cls[i].item())
                    conf     = float(boxes.conf[i].item())
                    cls_name = zone_model.names[c_id]

                    if conf < get_conf_threshold(cls_name):
                        continue

                    xyxy     = boxes.xyxy[i].cpu().numpy().astype(int).tolist()

                    # Polygon points from segmentation masks (xy scaled to frame)
                    pts = None
                    if masks is not None:
                        try:
                            # masks.xy gives a list of (N,2) arrays in pixel coords
                            xy = masks.xy[i]
                            if xy is not None and len(xy) >= 3:
                                pts = xy.astype(int).tolist()
                        except Exception:
                            pts = None

                    segs_to_save.append({
                        "xyxy": xyxy,
                        "conf": conf,
                        "name": cls_name,
                        "pts" : pts,
                    })

                    norm_name = cls_name.lower().replace(" ", "_")
                    for z_name in ALL_ZONES:
                        if YOLO_CLASS_MAP[z_name] == norm_name and conf > best_conf:
                            best_conf    = conf
                            detected_zone = z_name
                            detected_zone_xyxy = xyxy
                            break

            with cp_frame_lock:
                global latest_yolo_segs
                latest_yolo_segs = segs_to_save

            now = time.time()

            # ── Empty Table Detection ──────────────────────────────────────
            # Only triggers when YOLO explicitly detects the 'Empty_table' class
            # AND does not detect 'Part_OnThe_Table' (Person and Hand are ignored)
            # for 3 continuous seconds during an active running cycle.
            is_empty_table_detected = any(
                seg["name"].lower().replace(" ", "_") == "empty_table"
                for seg in segs_to_save
            )
            is_part_detected = any(
                seg["name"].lower().replace(" ", "_") == "part_onthe_table"
                for seg in segs_to_save
            )
            valid_empty_table = is_empty_table_detected and not is_part_detected

            with lock:
                cycle_result       = current_cycle.get("cycle_result")
                cycle_is_running   = (cycle_result == "running")
                cycle_is_finished  = (cycle_result in ("PASS", "FAIL"))
                is_auto_finishing  = current_cycle.get("auto_finishing", False)
                serial_ok          = current_cycle.get("serial_finalized", False)
                all_zones_done_chk = all(current_cycle.get(f"zone_{z}_status") == "done" for z in ALL_ZONES)

            # Evaluate empty table if cycle is running or finished
            if (cycle_is_running and not is_auto_finishing and not (all_zones_done_chk and serial_ok)) or cycle_is_finished:
                with lock:
                    has_progress = any(current_cycle.get(f"zone_{z}_status") == "done" for z in ALL_ZONES) or (current_cycle.get("serial") not in ("------", "", None))
                
                if valid_empty_table:
                    with lock:
                        if current_cycle.get("empty_table_first_seen") is None:
                            current_cycle["empty_table_first_seen"] = now
                            log.info("[EMPTY TABLE] 'Empty_table' class detected — starting 3s timer")
                        elapsed_empty = now - current_cycle["empty_table_first_seen"]
                    
                    empty_timeout = get_timing("empty_table_timeout", 3.0)
                    if elapsed_empty >= empty_timeout:
                        if cycle_is_finished:
                            log.info(f"[EMPTY TABLE] Part removed after completion — resetting cycle")
                            threading.Thread(target=_silent_reset_cycle, daemon=True).start()
                        elif has_progress:
                            log.info(f"[EMPTY TABLE] 'Empty_table' held for {elapsed_empty:.2f}s — triggering auto-finish")
                            threading.Thread(target=_force_finalize_no_serial, daemon=True).start()
                        else:
                            log.info(f"[EMPTY TABLE] {elapsed_empty:.2f}s with no progress — resetting silently (break time)")
                            threading.Thread(target=_silent_reset_cycle, daemon=True).start()
                else:
                    # Empty_table class not detected — reset the timer
                    with lock:
                        if current_cycle.get("empty_table_first_seen") is not None:
                            log.info("[EMPTY TABLE] 'Empty_table' class lost — resetting timer")
                        current_cycle["empty_table_first_seen"] = None
            else:
                with lock:
                    current_cycle["empty_table_first_seen"] = None

            # ── Update each zone's state machine ─────────────────────────────
            with lock:
                # OCR runs fully in parallel — no blocking of zone logic.
                # Zones and serial detection happen simultaneously.
                # Cycle only finalizes when BOTH all zones done AND serial finalized.

                expected_zone = _get_next_pending_zone()

                for zone in ALL_ZONES:
                    status = current_cycle[f"zone_{zone}_status"]
                    is_detected = (detected_zone == zone)
                    cfg = ZONE_CONFIG[zone]

                    # ── WAITING_FOR_ZONE: order-independent capture ───────
                    if status == "waiting_for_zone":
                        if is_detected:
                            # A pending zone is presented, begin capture!
                            current_cycle[f"zone_{zone}_status"] = "capturing"
                            zt[zone]["capture_start"] = now
                            zt[zone]["reference_xyxy"] = detected_zone_xyxy
                            if current_cycle["cycle_result"] == "idle":
                                current_cycle["cycle_result"] = "running"
                            capture_delay = get_timing(f"capture_delay_{zone}", 2.0)
                            current_cycle["instruction"] = get_banner("capturing", f"CAPTURING PHOTO TAKE HAND OUT ({capture_delay:.1f}s)", time=f"{capture_delay:.1f}")
                            current_cycle["instruction_color"] = "orange"
                        elif zone == expected_zone and not any(current_cycle[f"zone_{z}_status"] == "capturing" for z in ALL_ZONES):
                            # Only update the waiting instruction if no other zone is currently capturing
                            if detected_zone is not None and detected_zone in ALL_ZONES and current_cycle[f"zone_{detected_zone}_status"] == "done":
                                # Showing a zone that is already done
                                current_cycle["instruction"] = "ALREADY CAPTURED - SHOW NEXT ZONE"
                                current_cycle["instruction_color"] = "blue"
                            else:
                                current_cycle["instruction"] = get_banner("waiting", "WAITING FOR ANY ZONE", zone_name="ANY ZONE")
                                current_cycle["instruction_color"] = "blue"


                    # ── CAPTURING: steady capture window ──────────
                    elif status == "capturing":
                        capture_delay = get_timing(f"capture_delay_{zone}", 2.0)
                        elapsed = now - zt[zone].get("capture_start", now)
                        remaining = max(0, capture_delay - elapsed)

                        if remaining > 0:
                            if not is_detected:
                                # Zone lost too early — reset
                                current_cycle[f"zone_{zone}_status"] = "waiting_for_zone"
                                current_cycle["instruction"] = get_banner("lost_zone", f"LOST {cfg['name'].upper()} - PLEASE REPOSITION", zone_name=cfg['name'].upper())
                                current_cycle["instruction_color"] = "red"
                            else:
                                # Stability Check: ensure part is not rotating or moving
                                ref_box = zt[zone].get("reference_xyxy")
                                current_box = detected_zone_xyxy
                                if ref_box and current_box:
                                    cx1, cy1 = (ref_box[0]+ref_box[2])/2, (ref_box[1]+ref_box[3])/2
                                    cx2, cy2 = (current_box[0]+current_box[2])/2, (current_box[1]+current_box[3])/2
                                    dist = ((cx2-cx1)**2 + (cy2-cy1)**2)**0.5
                                    
                                    area1 = (ref_box[2]-ref_box[0]) * (ref_box[3]-ref_box[1])
                                    area2 = (current_box[2]-current_box[0]) * (current_box[3]-current_box[1])
                                    area_diff = abs(area1 - area2) / max(area1, 1)
                                    
                                    # If part moves > 80 pixels or area changes > 15%, it's unstable
                                    if dist > 80 or area_diff > 0.15:
                                        zt[zone]["capture_start"] = now
                                        zt[zone]["reference_xyxy"] = current_box
                                        elapsed = 0
                                        remaining = capture_delay
                                
                                current_cycle["instruction"] = get_banner("capturing", f"CAPTURING PHOTO TAKE HAND OUT ({remaining:.1f}s)", time=f"{remaining:.1f}")
                                current_cycle["instruction_color"] = "orange"
                        else:
                            # Capture window complete → check for hand/person in frame
                            is_hand_detected   = any(seg["name"].lower() == "hand"   for seg in segs_to_save)  # 'Hand'
                            is_person_detected = any(seg["name"].lower() == "person" for seg in segs_to_save)  # 'Person'

                            # For inner1: hand is allowed, but person still blocks.
                            # For all other zones: both hand and person block capture.
                            if zone == "inner1":
                                should_block = is_person_detected
                            else:
                                should_block = is_hand_detected or is_person_detected

                            if should_block:
                                zt[zone]["capture_start"] = now
                                blocker = "PERSON" if is_person_detected and not (is_hand_detected and zone != "inner1") else "HAND/PERSON"
                                current_cycle["instruction"] = get_banner("hand_detected", f"{blocker} DETECTED! PLEASE REMOVE FROM FRAME")
                                current_cycle["instruction_color"] = "red"
                            else:
                                # ── CAPTURE PHOTO ──
                                with cp_frame_lock:
                                    raw_to_save = cp_latest_frame.copy() if cp_latest_frame is not None else frame.copy()

                                _, clean_buf = cv2.imencode(".png", raw_to_save)
                                clean_jpeg = clean_buf.tobytes()
                                current_cycle.setdefault("zone_images", {})[zone] = clean_jpeg
                                log.info(f"[ZONE] {zone} CAPTURED clean frame → done (green)")
                                _save_image_to_disk(zone, clean_jpeg)

                                # ── Photo saved → immediately go green (done) ──
                                current_cycle[f"zone_{zone}_status"] = "done"
                                current_cycle[f"zone_{zone}_progress"] = 100

                                nxt = _get_next_pending_zone()
                                if nxt:
                                    current_cycle["instruction"] = "PHOTO SAVED! WAITING FOR NEXT ZONE"
                                    current_cycle["instruction_color"] = "green"
                                else:
                                    if not current_cycle.get("serial_finalized"):
                                        current_cycle["instruction"] = "ALL ZONES CAPTURED - WAITING FOR SERIAL"
                                        current_cycle["instruction_color"] = "orange"
                                _maybe_finalize_cycle()

                    # done → skip
        except Exception as e:
            log.error(f"Zone inference error: {e}")

        time.sleep(0.01)  # Run as fast as possible for smooth video

threading.Thread(target=zone_inference_loop, daemon=True).start()

def _silent_reset_cycle():
    """Silently resets the cycle without saving anything (used for false starts/break time)."""
    with lock:
        for z in ALL_ZONES:
            current_cycle[f"zone_{z}_status"]   = "waiting_for_zone"
            current_cycle[f"zone_{z}_progress"] = 0
            if zt:
                zt[z] = {
                    "status": "pending",
                    "accumulated": 0.0,
                    "detect_start": 0.0,
                    "capture_start": 0.0,
                    "last_seen": 0.0
                }
        current_cycle["cycle_result"]      = "idle"
        current_cycle["traceability_done"] = False
        current_cycle["serial"]            = "------"
        current_cycle["serial_date"]       = "------"
        current_cycle["serial_shift"]      = "-"
        current_cycle["serial_count"]      = "---"
        current_cycle["serial_time"]       = "--:--"
        current_cycle["confidence"]        = "- -"
        current_cycle["serial_finalized"]  = False
        current_cycle["ocr_phase"]         = "scanning"
        current_cycle["ocr_scan_start_time"] = time.time()
        current_cycle["holes_count"]       = 0
        current_cycle["defects"]           = []
        current_cycle["instruction"]       = "WAITING FOR PART"
        current_cycle["instruction_color"] = "blue"
        current_cycle["zone_images"]       = {}
        current_cycle.pop("ocr_raw_crop", None)
        current_cycle.pop("final_dir", None)
        current_cycle.pop("folder_name", None)
        current_cycle["folder_created"]          = False
        current_cycle["empty_table_first_seen"]  = None
        current_cycle["auto_finishing"]          = False
    # Tell OCR process to reset
    global current_ocr_box, ocr_reset_flag
    current_ocr_box = None
    ocr_reset_flag = True
    log.info("[BREAK TIME] False start detected. Cycle reset silently.")

def _force_finalize_no_serial():
    """
    Force-finalise the current cycle when the table becomes empty (operator
    removed the part without showing serial). Called from a background thread.

    - Saves whatever zone images were already captured.
    - Generates a PDF report marked FAIL with serial = NO_SERIAL.
    - Resets for the next cycle.
    """
    with lock:
        # Guard: only act if we're still in a running cycle that hasn't finished
        if current_cycle.get("cycle_result") in ("PASS", "FAIL"):
            return
        if current_cycle.get("auto_finishing"):
            return
        # Only trigger if cycle was actually active (at least one zone captured)
        any_zone_done = any(current_cycle.get(f"zone_{z}_status") == "done" for z in ALL_ZONES)
        if not any_zone_done and current_cycle.get("cycle_result") != "running":
            return

        current_cycle["auto_finishing"] = True
        current_cycle["instruction"] = "EMPTY TABLE DETECTED — AUTO SAVING CYCLE"
        current_cycle["instruction_color"] = "orange"

    log.info("[EMPTY TABLE] Operator removed part — auto-finalising cycle without serial.")
    time.sleep(0.4)  # brief pause so UI shows the message

    with lock:
        # Mark all un-captured zones as skipped (leave captured ones as 'done')
        for z in ALL_ZONES:
            if current_cycle[f"zone_{z}_status"] != "done":
                current_cycle[f"zone_{z}_status"] = "waiting_for_zone"  # leave idle, not error

        # Treat serial as absent
        current_cycle["serial_finalized"] = True  # force finalization path
        if current_cycle["serial"] in ("------", "", None):
            current_cycle["serial"] = "NO_SERIAL"

        serial_str = current_cycle["serial"]
        
        if serial_str == "NO_SERIAL":
            final_defects = ["Serial not captured — part removed from table"]
        else:
            final_defects = ["Incomplete inspection — part removed from table before all zones captured"]

        result = "FAIL"
        current_cycle["fail_cycles"] += 1
        current_cycle["total_cycles"] += 1
        current_cycle["cycle_result"]  = result

        try:
            cn = str(current_cycle["cycle_number"])
            num = int(cn.lstrip("#")) + 1
        except ValueError:
            num = 1
        current_cycle["cycle_number"] = f"#{num:03d}"

        log.info(f"[CYCLE] Auto-finalised (no serial/empty table) → FAIL  "
                 f"(total={current_cycle['total_cycles']}  "
                 f"pass={current_cycle['pass_cycles']}  "
                 f"fail={current_cycle['fail_cycles']})")

        # ── File saving & PDF ──
        today_str  = datetime.now().strftime("%Y-%m-%d")
        time_str   = datetime.now().strftime("%H:%M:%S")

        # Force folder name to timestamp_no_serial NOW (before serial is overwritten)
        if not current_cycle.get("folder_created"):
            timestamp_str = datetime.now().strftime("%H%M%S")
            day_dir = os.path.join(DATA_DIR, today_str)
            os.makedirs(day_dir, exist_ok=True)
            folder_name = f"{timestamp_str}_no_serial"
            final_dir   = os.path.join(day_dir, folder_name)
            os.makedirs(final_dir, exist_ok=True)
            current_cycle["final_dir"]      = final_dir
            current_cycle["folder_name"]    = folder_name
            current_cycle["folder_created"] = True
        else:
            final_dir, folder_name = _get_or_create_cycle_dir()

        # (Removed logic that renames folder to _no_image to prevent incorrect image file names)

        final_dir   = current_cycle["final_dir"]
        folder_name = current_cycle["folder_name"]

        # Confirm/write any zone images that were captured
        saved_images = {}
        for zone in ALL_ZONES:
            img_path = os.path.join(final_dir, f"{folder_name}_{zone}.png")
            if not os.path.exists(img_path):
                img_data = current_cycle.get("zone_images", {}).get(zone)
                if img_data is not None:
                    try:
                        if isinstance(img_data, bytes):
                            with open(img_path, "wb") as f:
                                f.write(img_data)
                        elif isinstance(img_data, np.ndarray):
                            cv2.imwrite(img_path, img_data)
                    except Exception as e:
                        log.error(f"[EMPTY TABLE] Failed to save {zone}: {e}")
            if os.path.exists(img_path):
                saved_images[zone] = img_path
                log.info(f"[IMAGE] Confirmed {zone} → {img_path}")

        crop_path = os.path.join(final_dir, f"{folder_name}_ocr.png")
        if not os.path.exists(crop_path) and current_cycle.get("ocr_raw_crop"):
            with open(crop_path, "wb") as f:
                f.write(current_cycle["ocr_raw_crop"])
        if os.path.exists(crop_path):
            saved_images["ocr_crop"] = crop_path
            log.info(f"[IMAGE] Confirmed OCR crop → {crop_path}")

        # Generate PDF report
        report_path = os.path.join(final_dir, f"{folder_name}.pdf")
        current_cycle["instruction"] = "CYCLE AUTO-SAVED (NO SERIAL) - NEXT PART"
        current_cycle["instruction_color"] = "red"

    # PDF generation outside the lock
    try:
        create_inspection_report(
            serial_number=serial_str,
            status="FAIL",
            date_str=today_str,
            time_str=time_str,
            zone_images=saved_images,
            ocr_serial=serial_str,
            confidence=None,
            defects=final_defects,
            output_path=report_path
        )
        log.info(f"[REPORT] Auto-finish report saved → {report_path}")
    except Exception as e:
        log.error(f"[REPORT] Failed to generate no-serial PDF: {e}")

    threading.Thread(target=_reset_after_delay, args=(2.0,), daemon=True).start()

def _maybe_finalize_cycle():
    """Called under lock. Finalises cycle when all zones + traceability done."""
    # Guard: Do not finalize if already finished or currently auto-finishing
    if current_cycle.get("cycle_result") in ("PASS", "FAIL") or current_cycle.get("auto_finishing"):
        return

    zones = [current_cycle[f"zone_{z}_status"] for z in ALL_ZONES]
    all_zones_done = all(s == "done" for s in zones)
    any_error = any(s == "error" for s in zones)
    trace_ok = current_cycle.get("serial_finalized", False)

    # If zones have errors, instruct operator to fix
    if any_error and all(s in ("done", "error") for s in zones):
        current_cycle["instruction"] = "ADJUST PART - ZONES MISSING"
        current_cycle["instruction_color"] = "orange"
        return

    # If all zones done but OCR not captured yet, instruct to keep part
    if all_zones_done and not trace_ok:
        current_cycle["instruction"] = "OCR NOT CAPTURED - KEEP PART FOR SERIAL"
        current_cycle["instruction_color"] = "orange"
        return

    # Need both zones and traceability to finalize
    if not (all_zones_done and trace_ok):
        return


    serial_bad = current_cycle["serial"] in ("------", "", None)

    if serial_bad:
        result = "FAIL"
        current_cycle["fail_cycles"] += 1
        current_cycle["instruction"] = "REMOVE PART - TRACEABILITY FAILED"
        current_cycle["instruction_color"] = "red"
    else:
        result = "PASS"
        current_cycle["pass_cycles"] += 1
        current_cycle["instruction"] = "CYCLE COMPLETE - REMOVE PART"
        current_cycle["instruction_color"] = "green"

    current_cycle["total_cycles"] += 1
    current_cycle["cycle_result"]  = result

    try:
        cn = str(current_cycle["cycle_number"])
        num = int(cn.lstrip("#")) + 1
    except ValueError:
        num = 1
    current_cycle["cycle_number"] = f"#{num:03d}"

    log.info(f"[CYCLE] Finalised → {result}  (total={current_cycle['total_cycles']}  "
             f"pass={current_cycle['pass_cycles']}  fail={current_cycle['fail_cycles']})")
             
    # --- File Saving & PDF Generation Logic ---
    today_str = datetime.now().strftime("%Y-%m-%d")
    timestamp_str = datetime.now().strftime("%H%M%S")
    time_str = datetime.now().strftime("%H:%M:%S")
    
    final_dir, folder_name = _get_or_create_cycle_dir()
    
    # (Removed defect_ folder prefix logic as requested by user)

    # Ensure all zone images and ocr crop are written to disk
    saved_images = {}
    for zone in ALL_ZONES:
        img_path = os.path.join(final_dir, f"{folder_name}_{zone}.png")
        if not os.path.exists(img_path):
            img_data = current_cycle.get("zone_images", {}).get(zone)
            if img_data is not None:
                if isinstance(img_data, bytes):
                    with open(img_path, "wb") as f:
                        f.write(img_data)
                elif isinstance(img_data, np.ndarray):
                    cv2.imwrite(img_path, img_data)
        if os.path.exists(img_path):
            saved_images[zone] = img_path
            log.info(f"[IMAGE] Confirmed {zone} → {img_path}")

    crop_path = os.path.join(final_dir, f"{folder_name}_ocr.png")
    if not os.path.exists(crop_path) and current_cycle.get("ocr_raw_crop"):
        with open(crop_path, "wb") as f:
            f.write(current_cycle["ocr_raw_crop"])
    if os.path.exists(crop_path):
        saved_images["ocr_crop"] = crop_path
        log.info(f"[IMAGE] Confirmed OCR crop → {crop_path}")

    # Generate PDF Report directly into final_dir
    report_path = os.path.join(final_dir, f"{folder_name}.pdf")
    serial_str = current_cycle.get("serial", "temp")
    try:
        create_inspection_report(
            serial_number=serial_str,
            status=result,
            date_str=today_str,
            time_str=time_str,
            zone_images=saved_images,
            ocr_serial=serial_str,
            confidence=None,
            defects=current_cycle.get("defects", []),
            output_path=report_path
        )
        log.info(f"[REPORT] Saved to {report_path}")
        # Only instruct operator to NEXT PART after PDF is fully generated
        current_cycle["instruction"] = "INSPECTION COMPLETE - REMOVE PART AND SCAN NEXT"
        current_cycle["instruction_color"] = "green" if result == "PASS" else "red"
    except Exception as e:
        log.error(f"[REPORT] Failed to generate PDF: {e}")
        current_cycle["instruction"] = "REPORT ERROR - REMOVE PART AND SCAN NEXT"
        current_cycle["instruction_color"] = "red"

    # NOTE: Removed automatic 3.5s reset thread. Cycle now stays active
    # and ONLY resets when empty_table_timeout detects the part was removed.

def _reset_after_delay(delay=2.5):
    """Pause so the UI can display PASS/FAIL, then reset for next cycle."""
    time.sleep(delay)
    with lock:
        for z in ALL_ZONES:
            current_cycle[f"zone_{z}_status"]   = "waiting_for_zone"
            current_cycle[f"zone_{z}_progress"] = 0
        current_cycle["cycle_result"]      = "idle"
        current_cycle["traceability_done"] = False
        current_cycle["serial"]            = "------"
        current_cycle["serial_date"]       = "------"
        current_cycle["serial_shift"]      = "-"
        current_cycle["serial_count"]      = "---"
        current_cycle["serial_time"]       = "--:--"
        current_cycle["confidence"]        = "- -"
        current_cycle["serial_finalized"]  = False
        current_cycle["ocr_phase"]         = "scanning"  # tells OCR process to start scanning
        current_cycle["ocr_scan_start_time"] = time.time()
        current_cycle["holes_count"]       = 0
        current_cycle["defects"]           = []
        current_cycle["instruction"]       = "WAITING FOR PART"
        current_cycle["instruction_color"] = "blue"
        current_cycle["zone_images"]       = {}
        current_cycle.pop("ocr_raw_crop", None)
        current_cycle.pop("final_dir", None)
        current_cycle.pop("folder_name", None)
        current_cycle["folder_created"]          = False
        current_cycle["empty_table_first_seen"]  = None
        current_cycle["auto_finishing"]          = False
    # Reset OCR voting state for the next cycle
    global current_ocr_box, ocr_reset_flag
    current_ocr_box = None
    ocr_reset_flag = True

# ── Standalone OCR Process routes ───────────────────────────────────────────────
@app.route("/latest_ocr_frame")
def latest_ocr_frame():
    """Returns the latest unannotated frame as JPEG for the OCR process."""
    global ind_latest_raw_frame, ocr_reset_flag
    if ind_latest_raw_frame is None:
        return jsonify({"status": "error", "message": "No frame available"}), 404
        
    _, buf = cv2.imencode(".jpg", ind_latest_raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    response = Response(buf.tobytes(), mimetype='image/jpeg')
    
    if ocr_reset_flag:
        response.headers["X-OCR-Reset"] = "1"
        ocr_reset_flag = False
        
    return response

@app.route("/update_ocr_box", methods=["POST"])
def update_ocr_box():
    """Receives the latest YOLO crop box from the OCR process."""
    global current_ocr_box
    data = request.get_json(silent=True) or {}
    box = data.get("box")
    if box and len(box) == 4:
        current_ocr_box = box
    return jsonify({"status": "success"})

# ── Zone Detection routes ─────────────────────────────────────────────────────
@app.route("/update_zone", methods=["POST"])
def update_zone():
    """
    Update a single zone's status/progress.
    Body: { "zone": "inner1"|"inner2"|"outer1"|"outer2",
            "status": "idle"|"detecting"|"done"|"error",
            "progress": 0-100 }
    """
    data    = request.get_json(silent=True) or {}
    zone    = data.get("zone",     "")
    status  = data.get("status",   "idle")
    progress = int(data.get("progress", 0))

    if zone not in ALL_ZONES:
        return jsonify({"status": "error",
                        "message": f"Unknown zone '{zone}'. Use: {ALL_ZONES}"}), 400

    with lock:
        current_cycle[f"zone_{zone}_status"]   = status
        current_cycle[f"zone_{zone}_progress"] = progress
        # Mark cycle running when first zone starts
        if status == "detecting" and current_cycle["cycle_result"] == "idle":
            current_cycle["cycle_result"] = "running"
        # Try to finalise if zone completed
        if status in ("done", "error"):
            _maybe_finalize_cycle()

    return jsonify({"status": "success"})


@app.route("/traceability_done", methods=["POST"])
def traceability_done_route():
    """
    Signal that OCR traceability captured a serial.
    Body: { "serial": "ABC123", "confidence": "98.5%", "finalized": true/false }
    The 'finalized' flag indicates the OCR voting has reached consensus.
    """
    data       = request.get_json(silent=True) or {}
    serial     = data.get("serial",     "------")
    confidence = data.get("confidence", "- -")
    finalized  = data.get("finalized",  False)

    date_p   = data.get("serial_date")
    shift_p  = data.get("serial_shift")
    count_p  = data.get("serial_count")
    time_p   = data.get("serial_time")
    if not date_p or not shift_p or not count_p:
        try:
            from ocr_processor import parse_serial_components
            date_p, shift_p, count_p, time_p, full_s = parse_serial_components(serial)
            if serial in ("------", "") and full_s != "------":
                serial = full_s
        except Exception:
            date_p, shift_p, count_p, time_p = "------", "-", "---", "--:--"

    with lock:
        current_cycle["traceability_done"] = True
        current_cycle["serial"]            = serial
        current_cycle["serial_date"]       = date_p
        current_cycle["serial_shift"]      = shift_p
        current_cycle["serial_count"]      = count_p
        current_cycle["serial_time"]       = time_p
        current_cycle["confidence"]        = confidence
        current_cycle["serial_finalized"]  = bool(finalized)
        # If finalized, extract the crop and update banner
        if finalized:
            global current_ocr_box
            # DO NOT set current_ocr_box to None here so the green tracking box stays visible
            if "raw_crop_base64" in data:
                import base64
                img_data = base64.b64decode(data["raw_crop_base64"])
                current_cycle["ocr_raw_crop"] = img_data
                _save_image_to_disk("ocr", img_data)

            current_cycle["ocr_phase"] = "completed"
            # Check if all zones are already done (serial came LAST)
            all_zones_done = all(current_cycle.get(f"zone_{z}_status") == "done" for z in ALL_ZONES)
            if all_zones_done:
                # Both serial and zones complete — finalize will be called below
                pass
            else:
                # Serial came FIRST — zones still pending
                current_cycle["instruction"] = "SERIAL LOCKED \u2713 - CONTINUE ZONE INSPECTION"
                current_cycle["instruction_color"] = "green"

        current_cycle["cycle_result"]      = current_cycle.get("cycle_result") or "running"
        _maybe_finalize_cycle()
    return jsonify({"status": "success", "serial": serial, "finalized": finalized})


@app.route("/reset_cycle", methods=["POST"])
def reset_cycle_route():
    """Manually resets the cycle and clears OCR."""
    global cycle_count, zt, ocr_reset_flag, current_ocr_box
    with lock:
        cycle_count += 1
        
        for z in ALL_ZONES:
            current_cycle[f"zone_{z}_status"]   = "waiting_for_zone"
            current_cycle[f"zone_{z}_progress"] = 0
            if zt:
                zt[z] = {
                    "status": "pending",
                    "accumulated": 0.0,
                    "detect_start": 0.0,
                    "capture_start": 0.0,
                    "last_seen": 0.0
                }
                
        current_cycle["is_processing"] = True
        current_cycle["status"] = "Cycle Reset"
        current_cycle["cycle_number"] = f"#{cycle_count:03d}"
        current_cycle["cycle_result"]      = "idle"
        current_cycle["traceability_done"] = False
        current_cycle["serial"]            = "------"
        current_cycle["serial_date"]       = "------"
        current_cycle["serial_shift"]      = "-"
        current_cycle["serial_count"]      = "---"
        current_cycle["serial_time"]       = "--:--"
        current_cycle["confidence"]        = "- -"
        current_cycle["serial_finalized"]  = False
        current_cycle["ocr_phase"]         = "scanning"
        current_cycle["ocr_scan_start_time"] = time.time()
        current_cycle["holes_count"]       = 0
        current_cycle["defects"]           = []
        current_cycle["instruction"]       = "WAITING FOR PART"
        current_cycle["instruction_color"] = "blue"
        current_cycle.pop("ocr_raw_crop", None)
        current_cycle.pop("final_dir", None)
        current_cycle.pop("folder_name", None)
        current_cycle["folder_created"]         = False
        current_cycle["empty_table_first_seen"] = None
        current_cycle["auto_finishing"]         = False
            
    # Reset OCR box mapping and tell standalone process to reset
    current_ocr_box = None
    ocr_reset_flag = True
            
    return jsonify({"status": "success", "message": "Cycle manually reset"})

# ── Status ────────────────────────────────────────────────────────────────────
@app.route("/status")
def status():
    with lock:
        resp = dict(current_cycle)
        # Deeply remove any bytes to prevent jsonify crashes
        keys_to_remove = [k for k, v in resp.items() if isinstance(v, bytes) or isinstance(v, dict)]
        for k in keys_to_remove:
            resp.pop(k, None)
    return jsonify(resp)

# ── Static assets (logos, etc.) ───────────────────────────────────────────────
@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(BASE_DIR, filename)

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  SHI Tail Gate Inspection Server")
    print("  CP Plus  ->  POST /connect_camera  {ip,port,user,pwd,path}")
    print("  GigE SDK ->  POST /connect_gige    {index,exposure,gain,...}")
    print("  Open UI  ->  http://localhost:5000")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
