# OCR Optimization Plan

I have reviewed the OCR pipeline in `ocr_processor.py`. Currently, it prioritizes extreme accuracy but does so in a way that sacrifices a lot of speed (taking more time than necessary). 

Here is how we can optimize it to be **fast and accurate, taking minimal time**:

## 1. Stop Running OCR Twice per Frame
**Current Issue:** The code currently runs the Heavy PaddleOCR neural network **twice** for every single frame (once on the raw crop, and once on the preprocessed crop). This cuts the speed exactly in half.
**Fix:** We will run PaddleOCR only *once* per frame on the pre-processed image (which generally yields the highest accuracy anyway).

## 2. Disable the Angle Classifier (Huge Speedup)
**Current Issue:** `PaddleOCR(use_angle_cls=True, ...)` runs a secondary AI model just to figure out if the text is upside down or rotated 90 degrees before reading it.
**Fix:** Since industrial serial numbers in a fixture are always right-side up and horizontal, we will set `use_angle_cls=False`. This eliminates an entire neural network pass, vastly improving speed with zero loss to accuracy.

## 3. Optimize the Denoising Filter
**Current Issue:** The code uses `cv2.bilateralFilter(d=9)` to remove noise. While accurate, bilateral filtering is notoriously slow on CPUs.
**Fix:** We will switch to a much faster `cv2.medianBlur` or `cv2.GaussianBlur`, or simply reduce the bilateral filter diameter from 9 to 5. This maintains edge definition but runs significantly faster.

## Open Questions

> [!IMPORTANT]
> 1. Is the serial number ever upside down or sideways when presented to the camera? (If it's always horizontal, turning off `use_angle_cls` is a free speed boost).
> 2. Are you comfortable with me applying these optimizations to make it faster while retaining the high accuracy?

## Proposed Changes

### `ocr_processor.py`
#### [MODIFY] [`ocr_processor.py`](file:///d:/SHI_TAILGATE_EDITED/ocr_processor.py)
- Change `use_angle_cls=True` to `False` in PaddleOCR initialization (line 171).
- Remove the `for img_variant, label in [(crop_frame, "raw"), (processed, "processed")]:` loop and only run OCR on `processed` (line 280).
- Optimize `cv2.bilateralFilter` by replacing it with `cv2.medianBlur(enhanced, 3)` (line 110).
