import numpy as np

def compute_ground_confidence(seg, depth, upto=250):
    """
    Heuristic ground safety confidence in [0,1].
    Higher is safer. Uses depth stability and obstacle density.
    - seg: binary road-like mask (1=road, 0=obstacle). If None, assume full frame.
    - depth: float32 HxW in [0,1] where larger is farther.
    """
    if depth is None or not isinstance(depth, np.ndarray):
        return 0.5

    h, w = depth.shape[:2]
    if isinstance(seg, np.ndarray) and seg.shape[:2] == depth.shape[:2]:
        road_mask = (seg > 0).astype(np.uint8)
    else:
        road_mask = np.ones((h, w), dtype=np.uint8)

    # Focus on near-field region (bottom 40% of the image)
    y0 = int(h * 0.6)
    nf_mask = np.zeros_like(road_mask)
    nf_mask[y0:, :] = 1
    mask = (road_mask & nf_mask).astype(bool)

    vals = depth[mask]
    if vals.size == 0:
        return 0.5

    # Metrics
    mean_d = float(np.nanmean(vals))
    std_d = float(np.nanstd(vals))
    # Estimate obstacle ratio as non-road in near-field
    obs_ratio = 1.0 - float(np.mean(road_mask[mask]))

    # Combine into a simple score: penalize high obstacle ratio and high variance (roughness)
    score = 1.0
    score -= 0.7 * np.clip(obs_ratio, 0, 1)
    # Normalize std by a typical scale (0.15) and cap
    score -= 0.5 * np.clip(std_d / 0.15, 0, 1)
    # If everything is too close (mean_d very small), reduce a bit
    score -= 0.2 * np.clip((0.2 - mean_d) / 0.2, 0, 1)

    return float(np.clip(score, 0.0, 1.0))
