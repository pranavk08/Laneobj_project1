import cv2
from typing import List, Optional, Tuple

Line = Tuple[int, int, int, int]


def _x_at_y(line: Line, y: int) -> float:
    x1, y1, x2, y2 = line
    if y1 == y2:
        return float(x1)
    t = (float(y) - y1) / (y2 - y1)
    return float(x1) + t * (x2 - x1)


def draw_output(
    img,
    lanes: List[Line],
    ground_conf: float,
    pixels_per_inch: Optional[float] = None,
    target_width_in: Optional[float] = None,
    pixels_per_foot: Optional[float] = None,
    lane_width_ft: Optional[float] = None,
):
    out = img.copy()
    color = (0, 255, 0) if ground_conf >= 0.5 else (0, 0, 255)
    
    # Draw filled lane area if we have 2 lanes
    if len(lanes) >= 2:
        h = out.shape[0]
        yt = int(0.6 * h)
        yb = h
        
        left, right = lanes[0], lanes[1]
        # Ensure left is left at bottom
        if _x_at_y(left, yb) > _x_at_y(right, yb):
            left, right = right, left
        
        # Create points for filled polygon
        pts = []
        # Left line from bottom to top
        x1_b, y1_b, x1_t, y1_t = left
        pts.append([x1_b, y1_b])
        pts.append([x1_t, y1_t])
        # Right line from top to bottom
        x2_b, y2_b, x2_t, y2_t = right
        pts.append([x2_t, y2_t])
        pts.append([x2_b, y2_b])
        
        # Draw semi-transparent fill
        import numpy as np
        overlay = out.copy()
        cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], (0, 255, 255))  # Yellow fill
        cv2.addWeighted(overlay, 0.3, out, 0.7, 0, out)
    
    # Draw lane lines on top
    for x1, y1, x2, y2 in lanes:
        cv2.line(out, (x1, y1), (x2, y2), color, 5)

    # Draw interior center lines if we have a scale in feet and lane width
    if pixels_per_foot and lane_width_ft and len(lanes) >= 2 and pixels_per_foot > 0:
        h = out.shape[0]
        yb, yt = h, int(0.6 * h)
        left, right = lanes[0], lanes[1]
        # ensure left is left at bottom
        if _x_at_y(left, yb) > _x_at_y(right, yb):
            left, right = right, left
        dx_px = max(0.0, _x_at_y(right, yb) - _x_at_y(left, yb))
        width_ft = dx_px / float(pixels_per_foot)
        # number of lanes across width (rounded to nearest int)
        n_lanes = int(max(1, round(width_ft / float(lane_width_ft))))
        # draw center lines at fractions
        for k in range(1, n_lanes):
            frac = k / n_lanes
            x_b = int(round(_x_at_y(left, yb) + frac * (_x_at_y(right, yb) - _x_at_y(left, yb))))
            x_t = int(round(_x_at_y(left, yt) + frac * (_x_at_y(right, yt) - _x_at_y(left, yt))))
            cv2.line(out, (x_b, yb), (x_t, yt), (0, 255, 255), 2)  # yellow center lines

    # Annotate measured lane/road width
    h = out.shape[0]
    yb = h
    if len(lanes) >= 2:
        left, right = lanes[0], lanes[1]
        if _x_at_y(left, yb) > _x_at_y(right, yb):
            left, right = right, left
        dx_px = max(0.0, _x_at_y(right, yb) - _x_at_y(left, yb))
        text = None
        if pixels_per_foot and pixels_per_foot > 0:
            width_ft = dx_px / float(pixels_per_foot)
            text = f"Road width: {width_ft:.2f} ft"
        elif pixels_per_inch and pixels_per_inch > 0:
            width_in = dx_px / float(pixels_per_inch)
            text = f"Road width: {width_in:.1f} in"
        if text is not None:
            cv2.putText(out, text, (20, out.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 215, 0), 2, cv2.LINE_AA)
        elif lane_width_ft:
            cv2.putText(out, "Set --pixels-per-foot to enable ft measurements", (20, out.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    return out
