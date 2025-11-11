import numpy as np
import cv2
from models.segmentation import Segmenter

# Lazy global segmenter
_SEGMENTER = None


def _get_segmenter():
    global _SEGMENTER
    if _SEGMENTER is None:
        try:
            _SEGMENTER = Segmenter(device="cuda")
        except Exception:
            _SEGMENTER = Segmenter(device="cpu")
    return _SEGMENTER


def _fast_fallback_segmentation(img):
    # Basic color/brightness heuristic for asphalt-like region
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    seg = ((gray > 50) & (gray < 200)).astype(np.uint8)
    return seg


def semantic_segmentation(img, depth, mode: str = "auto"):
    # Return a binary mask (1=road-like/drivable, 0=obstacle)
    if mode == "fast":
        seg = _fast_fallback_segmentation(img)
    else:
        seg = _get_segmenter().infer_road_mask(img)
    # Morphological cleanup
    seg = cv2.morphologyEx(seg.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    seg = (seg > 0).astype(np.uint8)
    # Restrict to trapezoidal ROI (focus on the road area) - consistent with road_model
    h, w = seg.shape[:2]
    mask = np.zeros_like(seg)
    roi = np.array([[
        (int(0.1 * w), h),                    # bottom left
        (int(0.4 * w), int(0.65 * h)),        # top left
        (int(0.6 * w), int(0.65 * h)),        # top right
        (int(0.9 * w), h)                     # bottom right
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi, 1)
    seg = (seg & mask).astype(np.uint8)
    # Optional: constrain by depth if available (treat very far as non-road)
    if depth is not None and isinstance(depth, np.ndarray) and depth.shape[:2] == seg.shape:
        # depth is normalized 0..1 in our DepthEstimator; treat far (>0.9) as non-road
        far = (depth > 0.9).astype(np.uint8)
        seg = (seg & (1 - far)).astype(np.uint8)
    return seg
