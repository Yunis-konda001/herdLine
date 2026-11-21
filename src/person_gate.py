"""Cheap person check so people-only clips are not counted as goats."""
from __future__ import annotations

import cv2
import numpy as np

# Global variables to cache the HOG model (so we only load it once).
_HOG = None
_HOG_UNAVAILABLE = False


def count_people_rough(frame: np.ndarray) -> int:
    """Return approximate number of standing people in a frame (OpenCV HOG).

    Returns 0 if HOG is unavailable in this OpenCV build.
    """
    global _HOG, _HOG_UNAVAILABLE
    
    # If HOG is unavailable or the frame is empty, return 0 people.
    if _HOG_UNAVAILABLE or frame is None or frame.size == 0:
        return 0
    
    try:
        # Load the HOG (Histogram of Oriented Gradients) person detector (only once).
        if _HOG is None:
            if not hasattr(cv2, "HOGDescriptor"):
                _HOG_UNAVAILABLE = True
                return 0
            hog = cv2.HOGDescriptor()
            # Use OpenCV's pre-trained people detector.
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            _HOG = hog
    except Exception:
        _HOG_UNAVAILABLE = True
        return 0

    # Resize large frames to 640px to speed up detection.
    h, w = frame.shape[:2]
    small = frame
    if max(h, w) > 640:
        scale = 640.0 / max(h, w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
    
    try:
        # Run the HOG detector to find people in the frame.
        rects, weights = _HOG.detectMultiScale(
            small,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
    except Exception:
        return 0
    
    # If no people were found, return 0.
    if weights is None or len(rects) == 0:
        return 0
    
    # Count how many detections had a strong confidence score (>= 0.4).
    n = 0
    for _rect, weight in zip(rects, weights):
        score = float(weight[0]) if hasattr(weight, "__len__") else float(weight)
        if score >= 0.4:
            n += 1
    return n