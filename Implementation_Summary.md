# Daily Implementation Summary

This document summarizes all the backend, UI, and ML updates implemented to the SHI Tail Gate Inspection Server today.

---

### 1. Strict Zone Inspection Sequencing
- **Implementation**: Enforced a hard sequence for the inspection pipeline (`Inner Zone 1` → `Inner Zone 2` → `Outer Zone 1` → `Outer Zone 2`).
- **Behavior**: If an operator touches a zone out of sequence (e.g., touching `Outer Zone 1` before finishing `Inner Zone 1`), the progress bar completely ignores it. The system flashes an orange warning banner explicitly directing the user to the correct zone (e.g., `"PLEASE INSPECT INNER ZONE 1 FIRST"`).

### 2. Dedicated 5-Second Capture Delay
- **Implementation**: Separated the "progress" phase from the "capture" phase inside the `app.py` state machine by updating the `CAPTURE_DELAY` to `5.0`.
- **Behavior**: Once a zone's progress bar hits 100%, the backend triggers a dedicated 5-second countdown. The main banner updates to `"CAPTURING PICTURE KEEP PART CORRECT AND TAKE HAND OUT"`. The clean frame is only captured *after* this 5-second window has passed, ensuring no hands are caught in the image.

### 3. Extended 5-Image PDF Report
- **Implementation**: Overhauled the `generate_report.py` canvas script to fit 5 images on a single A4 page. 
- **Behavior**: The backend now dynamically scans the `tailgate_data/<serial>/crop/` folder to pull the original raw OCR serial crop. This raw crop is now proudly displayed at the top center of the generated PDF, followed by a 2x2 grid of the four zone images below it.

### 4. Progress Time Adjustments
- **Implementation**: Modified the `ZONE_CONFIG` variables inside `app.py`.
- **Behavior**: 
  - `Inner Zone 1` and `Outer Zone 1` now take exactly **15 seconds** to progress.
  - `Inner Zone 2` and `Outer Zone 2` take **8 seconds**.

### 5. UI Serial Number Formatting
- **Implementation**: Added regex parsing logic to the `pollStatus()` loop in `index.html`.
- **Behavior**: 14-character serial numbers are now broken apart in the browser for extreme readability. For example, `310826A1511204` now beautifully renders as `310826 A151 12:04`. The raw unformatted string is still safely preserved in the backend for accurate folder/PDF generation.

### 6. YOLOv2 Model Integration
- **Implementation**: Upgraded the hardcoded paths for the AI inference engines.
- **Behavior**: 
  - The 4-zone detection logic now uses the `shi-seq-v2-100-epochs.pt` model.
  - The OCR region detection logic now uses the `shi-serial-v2.pt` model.
