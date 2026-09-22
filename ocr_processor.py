import cv2
import threading
import queue
import time
import requests
import logging
import re
import os
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

def parse_serial_components(raw_text):
    """
    Split serial text into Date, Model, Time, and Full serial.
    Expected format: 14 chars -> Date (6, DDMMYY) + Model (4) + Time (4, HHMM)
    e.g. '170926 A 065 09:47' or '170926A0650947'
    Returns: (date_str, model_str, time_str, full_clean_str)
    """
    if not raw_text or raw_text in ("------", ""):
        return "------", "----", "--:--", "------"

    clean = re.sub(r'[^A-Za-z0-9]', '', raw_text)

    # 1. Match spaced/colon format: e.g. "170926 A 065 09:47"
    m = re.search(r'(\d{6})\s*([A-Za-z]\s*\d{3})\s*(\d{2})[:.]?(\d{2})', raw_text)
    if m:
        date_p = m.group(1)
        model_p = re.sub(r'\s+', '', m.group(2)).upper()
        time_p = f"{m.group(3)}:{m.group(4)}"
        full_s = f"{date_p}{model_p}{m.group(3)}{m.group(4)}"
        return date_p, model_p, time_p, full_s

    # 2. Exactly 14 alphanumeric chars: 6 Date + 4 Model + 4 Time
    if len(clean) == 14:
        date_p = clean[:6]
        model_p = clean[6:10].upper()
        time_p = f"{clean[10:12]}:{clean[12:14]}"
        return date_p, model_p, time_p, clean

    # 3. Partial reads (10 to 13 chars)
    if len(clean) >= 10:
        date_p = clean[:6]
        model_p = clean[6:10].upper()
        rem = clean[10:]
        time_p = f"{rem[:2]}:{rem[2:]}" if len(rem) == 4 else rem
        return date_p, model_p, time_p, clean

    return clean, "----", "--:--", clean

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
    Apply a pipeline of image processing techniques to maximize OCR accuracy:
      1. Resize (2x zoom) for better character resolution
      2. Convert to grayscale
      3. CLAHE (Contrast Limited Adaptive Histogram Equalization) for local contrast
      4. Bilateral filter to remove noise while preserving edges
      5. Otsu thresholding for binarization
      6. Morphological close to fill small gaps in characters
    Returns: processed image (grayscale, uint8)
    """
    if crop_img is None or crop_img.size == 0:
        return crop_img

    # 1. Zoom: upscale 2x with cubic interpolation for sub-pixel detail
    h, w = crop_img.shape[:2]
    zoomed = cv2.resize(crop_img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # 2. Grayscale
    if len(zoomed.shape) == 3:
        gray = cv2.cvtColor(zoomed, cv2.COLOR_BGR2GRAY)
    else:
        gray = zoomed.copy()

    # 3. CLAHE – adaptive contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 4. Median blur – fast noise reduction while preserving edges
    denoised = cv2.medianBlur(enhanced, 3)

    # 5. Otsu threshold – automatic binarization
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 6. Morphological close – fill tiny gaps in character strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return cleaned


def sharpen_image(img):
    """Apply an unsharp mask for additional sharpening before OCR."""
    gaussian = cv2.GaussianBlur(img, (0, 0), 3)
    sharpened = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    return sharpened


# ── Crop Saving Helper (Removed) ──────────────────────────────────────────────
# We no longer save crops to disk here. The crop is passed to the main app via base64.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class TailgateOCR(threading.Thread):
    def __init__(self, endpoint_url="http://127.0.0.1:5000/traceability_done",
                 vote_threshold=2, vote_window=3):
        """
        Initializes the OCR thread with voting mechanism.
        
        Args:
            endpoint_url: Flask endpoint to report finalized serial
            vote_threshold: minimum identical readings to finalize (default: 2 out of 3)
            vote_window: number of recent readings to consider (default: 3)
        """
        super().__init__(daemon=True)
        self.endpoint_url = endpoint_url
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True
        
        # ── Voting state ──────────────────────────────────────────────────────
        self.vote_threshold = vote_threshold
        self.vote_window = vote_window
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
            
            model_path = os.path.join(BASE_DIR, "Models", "SHI_SERIAL_V3.pt")
            log.info(f"Initializing YOLO ({model_path}) for OCR region detection...")
            self.yolo_model = YOLO(model_path)
            
            log.info("Initializing PaddleOCR (Running on CPU to prevent CUDA DLL conflicts with YOLO)...")
            self.ocr = PaddleOCR(lang='en')
            log.info("AI Models loaded successfully.")
        except Exception as e:
            log.error(f"Failed to initialize OCR models: {e}")
            self.ocr = None
            self.yolo_model = None

    def process_frame(self, frame):
        """
        Non-blocking function to add a frame to the processing queue.
        If the queue is full (OCR is currently busy), the frame is dropped to prevent getting stuck.
        Call this from your main camera loop.
        """
        # If serial is already finalized for this cycle, skip entirely
        with self._vote_lock:
            if self.finalized_serial is not None:
                return
        try:
            # We copy the frame so the camera thread can reuse the original buffer if needed
            self.frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass # Drop frame to avoid blocking the camera feed

    def reset_voting(self):
        """Reset voting state for a new cycle (called when cycle resets)."""
        with self._vote_lock:
            self.recent_readings.clear()
            self.finalized_serial = None
            self.has_reported = False
            self.frame_counter = 0
            log.info("[VOTE] Voting state reset for new cycle.")

    def run(self):
        log.info("Tailgate OCR Thread Started (with voting + image processing). Waiting for frames...")
        while self.running:
            try:
                # Wait for a frame with a timeout so we can gracefully exit if running = False
                frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                self.perform_ocr(frame)
            except Exception as e:
                log.error(f"Error during OCR processing: {e}")
            finally:
                self.frame_queue.task_done()

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
            results = self.yolo_model(frame, verbose=False)
        except Exception as e:
            log.error(f"YOLO inference error: {e}")
            return
            
        # Check if there are any detections
        if len(results) == 0 or len(results[0].boxes) == 0:
            # If YOLO doesn't detect anything, return to prevent false positives
            return

        # Get the box with the highest confidence
        box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        
        # Add padding around detected region for better OCR context
        h, w = frame.shape[:2]
        pad = 15  # slightly more padding than before for zoom
        y1 = max(0, y1 - pad)
        y2 = min(h, y2 + pad)
        x1 = max(0, x1 - pad)
        x2 = min(w, x2 + pad)
        
        crop_frame = np.ascontiguousarray(frame[y1:y2, x1:x2])
        
        if crop_frame.size == 0:
            return

        # ── Step 2: Image preprocessing pipeline ─────────────────────────────
        # First sharpen the raw crop for better edge definition
        sharpened_crop = sharpen_image(crop_frame)
        
        # Then run full preprocessing (zoom, CLAHE, bilateral, Otsu, morphology)
        processed = preprocess_for_ocr(sharpened_crop)

        # ── Step 3: Run PaddleOCR on BOTH raw crop and processed version ─────
        # We run ONLY on the processed image for maximum speed and accuracy
        best_text = ""
        best_conf = 0.0
        best_score = -1.0
        
        try:
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
                self.report_traceability(serial_to_report, confidence_str, finalized=True, crop_frame=crop_frame)

    def report_traceability(self, serial, confidence, finalized=False, crop_frame=None):
        date_p, model_p, time_p, full_s = parse_serial_components(serial)
        payload = {
            "serial": full_s if full_s != "------" else serial,
            "serial_date": date_p,
            "serial_model": model_p,
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
        self.join()

# --- Example Usage / Testing ---
if __name__ == "__main__":
    ocr_worker = TailgateOCR()
    ocr_worker.start()
    
    log.info("Simulating camera feed...")
    try:
        while True:
            dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
            cv2.putText(dummy_frame, "SHI-88942", (100, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
            
            ocr_worker.process_frame(dummy_frame)
            time.sleep(0.1) 
    except KeyboardInterrupt:
        log.info("Stopping...")
        ocr_worker.stop()
