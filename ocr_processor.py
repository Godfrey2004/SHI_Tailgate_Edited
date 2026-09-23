import cv2
import threading
import time
import requests
import logging
import re
import os
import yaml
import numpy as np
from collections import Counter
from datetime import datetime

# Pre-load PyTorch/YOLO to initialize C++ runtime and prevent Windows DLL conflicts with Paddle
try:
    from ultralytics import YOLO
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [OCR] %(message)s")
log = logging.getLogger("OCR")

# ── Load config.yaml ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_cfg = {}
try:
    with open(os.path.join(BASE_DIR, "config.yaml"), "r") as f:
        _cfg = yaml.safe_load(f) or {}
    log.info("Loaded config.yaml for OCR settings.")
except Exception as e:
    log.warning(f"Could not load config.yaml, using defaults: {e}")

_ocr_cfg = _cfg.get("ocr", {})

# ── OCR tuning parameters (all from config.yaml) ──────────────────────────────
OCR_VOTE_WINDOW     = int(_ocr_cfg.get("vote_window",    4))
OCR_VOTE_THRESHOLD  = int(_ocr_cfg.get("vote_threshold", 2))

# Image filter flags
OCR_ZOOM_FACTOR          = int(_ocr_cfg.get("zoom_factor",       2))
OCR_CLAHE_ENABLED        = bool(_ocr_cfg.get("clahe_enabled",    True))
OCR_CLAHE_CLIP           = float(_ocr_cfg.get("clahe_clip_limit", 3.0))
OCR_CLAHE_TILE           = int(_ocr_cfg.get("clahe_tile_size",    8))
OCR_MEDIAN_BLUR_ENABLED  = bool(_ocr_cfg.get("median_blur_enabled", True))
OCR_MEDIAN_BLUR_KSIZE    = int(_ocr_cfg.get("median_blur_ksize",    3))
OCR_OTSU_ENABLED         = bool(_ocr_cfg.get("otsu_enabled",        True))
OCR_MORPH_CLOSE_ENABLED  = bool(_ocr_cfg.get("morph_close_enabled", True))
OCR_MORPH_CLOSE_KSIZE    = int(_ocr_cfg.get("morph_close_ksize",    2))
OCR_SHARPEN_ENABLED      = bool(_ocr_cfg.get("sharpen_enabled",     True))
OCR_AUTO_ROTATE_ENABLED  = bool(_ocr_cfg.get("auto_rotate_enabled", True))

log.info(f"OCR Config: vote_window={OCR_VOTE_WINDOW}, vote_threshold={OCR_VOTE_THRESHOLD}, "
         f"zoom={OCR_ZOOM_FACTOR}x, clahe={OCR_CLAHE_ENABLED}, blur={OCR_MEDIAN_BLUR_ENABLED}, "
         f"otsu={OCR_OTSU_ENABLED}, morph={OCR_MORPH_CLOSE_ENABLED}, "
         f"sharpen={OCR_SHARPEN_ENABLED}, auto_rotate={OCR_AUTO_ROTATE_ENABLED}")

def parse_serial_components(raw_text):
    """
    Split serial text into Date, Shift, Count, Time, and Full serial.
    Expected format: 14 chars -> Date (6 digits) + Shift (1 letter) + Count (3 digits) + Time (4 digits)
    Returns: (date_str, shift_str, count_str, time_str, full_clean_str)
    """
    if not raw_text or raw_text in ("------", ""):
        return "------", "-", "---", "--:--", "------"

    # Strip all non-alphanumeric just to have a clean base
    clean = re.sub(r'[^A-Za-z0-9]', '', raw_text)
    
    # We will pad it to 14 chars with spaces if it's too short just to avoid index errors
    clean_padded = clean.ljust(14, ' ')

    # 1. Date (first 6 chars) -> Must be 6 digits. Fallback to system date if not.
    raw_date = clean_padded[:6]
    date_p = re.sub(r'[^0-9]', '', raw_date)
    if len(date_p) < 6:
        date_p = datetime.now().strftime("%d%m%y")
    else:
        date_p = date_p[:6]
        
    # 2. Shift (7th char) -> Must be A, B, or C
    raw_shift = clean_padded[6].upper()
    shift_map = {'8': 'B', 'V': 'A', 'U': 'A', '0': 'C', 'O': 'C'}
    if raw_shift in shift_map:
        shift_p = shift_map[raw_shift]
    elif raw_shift in ('A', 'B', 'C'):
        shift_p = raw_shift
    else:
        shift_p = '-'
        
    # 3. Count (8th to 10th chars) -> Must be 3 digits
    raw_count = clean_padded[7:10]
    count_p = re.sub(r'[^0-9]', '', raw_count)
    if len(count_p) == 0:
        count_p = "---"
    else:
        count_p = count_p.zfill(3)[:3]
    
    # 4. Time (11th to 14th chars) -> Must be 4 digits
    raw_time = clean_padded[10:14]
    time_digits = re.sub(r'[^0-9]', '', raw_time)
    if len(time_digits) >= 4:
        time_p = f"{time_digits[:2]}:{time_digits[2:4]}"
    elif len(time_digits) > 0:
        time_p = time_digits
    else:
        time_p = "--:--"
        
    full_s = f"{date_p}{shift_p}{count_p}{time_digits[:4]}"
    return date_p, shift_p, count_p, time_p, full_s

def score_serial_candidate(text, conf):
    """
    Score OCR candidate to prevent truncated reads from overriding complete serials.
    A complete 14-char read (Date 6 + Model 4 + Time 4) gets top priority.
    """
    clean = re.sub(r'[^A-Za-z0-9]', '', text)
    if not clean or len(clean) < 5:
        return -1.0, ""
    
    is_full_14 = (len(clean) == 14)
    starts_with_date = bool(re.match(r'^\d{6}', clean))
    
    if is_full_14 and starts_with_date:
        score = 10.0 + conf
    elif len(clean) >= 10 and starts_with_date:
        score = (len(clean) / 14.0) * 2.0 + conf
    else:
        score = (len(clean) / 14.0) + conf
    return score, clean

# ── Image Processing Filters ─────────────────────────────────────────────────
def preprocess_for_ocr(crop_img):
    """
    Configurable preprocessing pipeline (controlled via config.yaml).
    Only the enabled filters run — disable any via config to save time.
    """
    if crop_img is None or crop_img.size == 0:
        return crop_img

    # 1. Zoom: upscale for better character resolution (ESSENTIAL)
    h, w = crop_img.shape[:2]
    zoomed = cv2.resize(crop_img, (w * OCR_ZOOM_FACTOR, h * OCR_ZOOM_FACTOR), interpolation=cv2.INTER_CUBIC)

    # 2. Grayscale (always required)
    gray = cv2.cvtColor(zoomed, cv2.COLOR_BGR2GRAY) if len(zoomed.shape) == 3 else zoomed.copy()

    # 3. CLAHE - adaptive contrast (config: clahe_enabled)
    if OCR_CLAHE_ENABLED:
        clahe = cv2.createCLAHE(clipLimit=OCR_CLAHE_CLIP, tileGridSize=(OCR_CLAHE_TILE, OCR_CLAHE_TILE))
        gray = clahe.apply(gray)
    # else: skip CLAHE (faster, use if image already has good contrast)

    # 4. Median blur - noise reduction (config: median_blur_enabled)
    if OCR_MEDIAN_BLUR_ENABLED:
        gray = cv2.medianBlur(gray, OCR_MEDIAN_BLUR_KSIZE)
    # else: skip blur (faster, use if camera image is already clean)

    # 5. Otsu binarization - black/white conversion (config: otsu_enabled)
    if OCR_OTSU_ENABLED:
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # else: pass grayscale directly (may help on some image types)

    # 6. Morphological close - fill broken character strokes (config: morph_close_enabled)
    if OCR_MORPH_CLOSE_ENABLED:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (OCR_MORPH_CLOSE_KSIZE, OCR_MORPH_CLOSE_KSIZE))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    # else: skip morph (faster, use if characters are already complete strokes)

    return gray


def sharpen_image(img):
    """Unsharp mask sharpening (config: sharpen_enabled)."""
    gaussian = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)


# ── Crop Saving Helper (Removed) ──────────────────────────────────────────────
# We no longer save crops to disk here. The crop is passed to the main app via base64.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class TailgateOCR:
    def __init__(self, endpoint_url="http://127.0.0.1:5000/traceability_done",
                 vote_threshold=None, vote_window=None):
        """
        Initializes the OCR standalone process.
        vote_threshold and vote_window default to config.yaml values.
        """
        self.endpoint_url = endpoint_url
        self.running = True
        
        # Voting state — read from config.yaml by default
        self.vote_threshold = vote_threshold if vote_threshold is not None else OCR_VOTE_THRESHOLD
        self.vote_window    = vote_window    if vote_window    is not None else OCR_VOTE_WINDOW
        self.recent_readings = []       # list of (serial_text, confidence) tuples
        self.finalized_serial = None    # once locked, stops OCR until reset
        self.has_reported = False       # ensures finalized serial is reported only once per cycle
        self.frame_counter = 0          # counts frames processed in this cycle
        self._vote_lock = threading.Lock()
        
        # Initialize PaddleOCR
        try:
            # IMPORTANT: Import YOLO (which loads PyTorch) BEFORE PaddleOCR 
            # to prevent Windows DLL loading conflicts between Torch and Paddle.
            from ultralytics import YOLO
            from paddleocr import PaddleOCR
            
            model_path = os.path.join(BASE_DIR, "Models", "serial", "best.pt")
            log.info(f"Initializing YOLO ({model_path}) for OCR region detection...")
            self.yolo_model = YOLO(model_path)
            
            log.info("Initializing PaddleOCR (Running on CPU to prevent CUDA DLL conflicts with YOLO)...")
            self.ocr = PaddleOCR(lang='en', use_angle_cls=True)
            log.info("AI Models loaded successfully.")
        except Exception as e:
            import traceback
            log.error(f"Failed to initialize OCR models: {e}\n{traceback.format_exc()}")
            self.ocr = None
            self.yolo_model = None
        self.latest_box = None

    def fetch_latest_frame(self):
        """Fetches the latest unannotated frame from the Flask server."""
        try:
            resp = requests.get("http://127.0.0.1:5000/latest_ocr_frame", timeout=1.0)
            if resp.status_code == 200:
                if resp.headers.get("X-OCR-Reset") == "1":
                    self.reset_voting()
                img_array = np.frombuffer(resp.content, dtype=np.uint8)
                return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except requests.exceptions.RequestException:
            pass
        return None

    def reset_voting(self):
        """Reset voting state for a new cycle (called when cycle resets)."""
        with self._vote_lock:
            self.recent_readings.clear()
            self.finalized_serial = None
            self.has_reported = False
            self.frame_counter = 0
            log.info("[VOTE] Voting state reset for new cycle.")

    def run(self):
        log.info("Tailgate OCR Standalone Process Started. Polling for frames...")
        while self.running:
            frame = self.fetch_latest_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            try:
                self.perform_ocr(frame)
            except Exception as e:
                log.error(f"Error during OCR processing: {e}")
            
            # Throttle slightly to not hammer the server if OCR finishes very quickly
            time.sleep(0.05)

    def perform_ocr(self, frame):
        """
        Full OCR pipeline:
          1. YOLO detection → crop region
          2. Image preprocessing (zoom, filters, binarize)
          3. PaddleOCR on processed image
          4. Voting across multiple frames
          5. Save crops to disk
          6. Report finalized serial when vote passes
        """
        if self.ocr is None or self.yolo_model is None:
            time.sleep(0.5)
            return

        # Skip if already finalized
        with self._vote_lock:
            if self.finalized_serial is not None:
                return

        # ── Step 1: YOLO detection to locate serial number region ─────────────
        try:
            results = self.yolo_model(frame, conf=0.15, verbose=False)
        except Exception as e:
            log.error(f"YOLO inference error: {e}")
            return
            
        # Check if there are any detections
        if len(results) == 0 or len(results[0].boxes) == 0:
            # If YOLO doesn't detect anything, return to prevent false positives
            self.latest_box = None
            return

        # Get the box with the highest confidence
        box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        self.latest_box = (x1, y1, x2, y2)
        
        # POST the box coordinates back to the main app so it can draw the green rectangle
        try:
            requests.post("http://127.0.0.1:5000/update_ocr_box", json={"box": [int(x1), int(y1), int(x2), int(y2)]}, timeout=0.5)
        except requests.exceptions.RequestException:
            pass
        
        # Add padding around detected region for better OCR context
        h, w = frame.shape[:2]
        pad = 15  # Reverted back to 15 to avoid background noise
        y1 = max(0, y1 - pad)
        y2 = min(h, y2 + pad)
        x1 = max(0, x1 - pad)
        x2 = min(w, x2 + pad)
        
        crop_frame = np.ascontiguousarray(frame[y1:y2, x1:x2])
        
        if crop_frame.size == 0:
            return

        # Save original crop to send to the UI (so UI doesn't look rotated)
        ui_crop = crop_frame.copy()

        # ── Step 2: Image Preprocessing pipeline ─────────────────────────────
        # Auto-rotate if crop is vertical (config: auto_rotate_enabled)
        ch, cw = crop_frame.shape[:2]
        if OCR_AUTO_ROTATE_ENABLED and ch > cw * 1.2:
            crop_frame = cv2.rotate(crop_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        # Sharpen before preprocessing (config: sharpen_enabled)
        if OCR_SHARPEN_ENABLED:
            crop_frame = sharpen_image(crop_frame)
        # else: skip sharpening (use if camera is already crisp)
        
        # Run the configurable preprocessing pipeline
        processed = preprocess_for_ocr(crop_frame)

        # ── Step 3: Run PaddleOCR ────────────────────────────────────────────
        best_text = ""
        best_conf = 0.0
        best_score = -1.0
        
        try:
            # Disable angle classifier (cls=False) for faster speed since we pre-rotated
            result = self.ocr.ocr(processed, cls=False)
        except Exception as e:
            log.warning(f"OCR failed on processed image: {e}")
            result = None
            
        if result and result[0]:
            detected_texts = []
            confidences = []
            for line in result[0]:
                if line and len(line) > 1:
                    text_info = line[1]
                    detected_texts.append(text_info[0])
                    confidences.append(text_info[1])
                    
            full_text = " ".join(detected_texts)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            score, cleaned = score_serial_candidate(full_text, avg_conf)
            
            if score > best_score:
                best_score = score
                best_text = cleaned
                best_conf = avg_conf
                log.info(f"  [PROCESSED] → '{cleaned}' (conf={avg_conf*100:.1f}%, score={score:.2f})")

        if not best_text:
            return

        # ── Step 4: Voting mechanism ─────────────────────────────────────────
        with self._vote_lock:
            self.frame_counter += 1
            frame_idx = self.frame_counter
            
            self.recent_readings.append((best_text, best_conf))
            
            # Keep only the last N readings
            if len(self.recent_readings) > self.vote_window:
                self.recent_readings = self.recent_readings[-self.vote_window:]
            
            log.info(f"[VOTE] Frame #{frame_idx}: '{best_text}' ({best_conf*100:.1f}%) "
                     f"| History: {[r[0] for r in self.recent_readings]}")
            
            # Check for majority vote
            if len(self.recent_readings) >= self.vote_threshold:
                serial_counts = Counter(r[0] for r in self.recent_readings)
                most_common_serial, count = serial_counts.most_common(1)[0]
                
                if count >= self.vote_threshold:
                    # ── FINALIZED! Lock the serial number ─────────────────────
                    self.finalized_serial = most_common_serial
                    
                    # Calculate average confidence across matching reads
                    matching_confs = [r[1] for r in self.recent_readings if r[0] == most_common_serial]
                    final_conf = sum(matching_confs) / len(matching_confs)
                    
                    log.info(f"[VOTE] ✓✓✓ SERIAL FINALIZED: '{most_common_serial}' "
                             f"({count}/{len(self.recent_readings)} votes, "
                             f"avg conf={final_conf*100:.1f}%)")
                else:
                    log.info(f"[VOTE] No consensus yet. Best: '{most_common_serial}' "
                             f"({count}/{self.vote_threshold} needed)")

        # ── Step 5: (Removed Disk Saving) ────────────────────────────────────
        # No local disk saving here to prevent duplicates.


        # ── Step 6: Report finalized serial to the main app ──────────────────
        with self._vote_lock:
            if self.finalized_serial is not None and not self.has_reported:
                self.has_reported = True
                serial_to_report = self.finalized_serial
                matching_confs = [r[1] for r in self.recent_readings if r[0] == serial_to_report]
                final_conf = sum(matching_confs) / len(matching_confs) if matching_confs else best_conf
                confidence_str = f"{final_conf*100:.1f}%"
        
                # Report with finalized=true flag so UI turns green, and send the clean frame
                self.report_traceability(serial_to_report, confidence_str, finalized=True, crop_frame=ui_crop)

    def report_traceability(self, serial, confidence, finalized=False, crop_frame=None):
        date_p, shift_p, count_p, time_p, full_s = parse_serial_components(serial)
        payload = {
            "serial": full_s if full_s != "------" else serial,
            "serial_date": date_p,
            "serial_shift": shift_p,
            "serial_count": count_p,
            "serial_time": time_p,
            "confidence": confidence,
            "finalized": finalized
        }
        if crop_frame is not None:
            try:
                import base64
                _, buffer = cv2.imencode('.jpg', crop_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                payload["raw_crop_base64"] = base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                log.warning(f"Failed to encode raw crop frame: {e}")
        try:
            # Short timeout to ensure we don't hang on network issues
            resp = requests.post(self.endpoint_url, json=payload, timeout=2.0)
            if resp.status_code == 200:
                status_tag = "FINALIZED ✓" if finalized else "interim"
                log.info(f"Successfully reported serial {serial} ({status_tag}) to main app.")
            else:
                log.warning(f"Failed to report serial. Server responded with {resp.status_code}")
        except requests.exceptions.RequestException as e:
            log.error(f"Connection error when reporting serial: {e}")

    def stop(self):
        self.running = False

# --- Standalone Execution ---
if __name__ == "__main__":
    # Ensure working directory is correct so models can be found
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    ocr_worker = TailgateOCR()
    try:
        ocr_worker.run()
    except KeyboardInterrupt:
        log.info("Stopping Standalone OCR Engine...")
        ocr_worker.stop()
