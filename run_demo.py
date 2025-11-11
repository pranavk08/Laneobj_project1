import argparse
import cv2
import numpy as np

from models.depth_model import DepthEstimator
from models import turn_predictor
from utils import preprocess, road_model, ground_model, visualize


def open_source(source):
    # Allow int (webcam) or string path (video file)
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    return cv2.VideoCapture(source)


def compute_curvature_proxy(lanes, h):
    # Simple curvature proxy: positive for right turn, negative for left
    if not lanes:
        return 0.0
    xs_bottom = []
    xs_top = []
    for x1, y1, x2, y2 in lanes:
        # Normalize to y increasing downward; bottom is y=h
        xb = x1 if y1 > y2 else x2
        xt = x2 if y1 > y2 else x1
        xs_bottom.append(xb)
        xs_top.append(xt)
    if not xs_bottom or not xs_top:
        return 0.0
    mean_bottom = float(np.mean(xs_bottom))
    mean_top = float(np.mean(xs_top))
    # Positive if lanes shift to the right as we go up (indicative of right curve)
    return (mean_top - mean_bottom) / max(1.0, float(h))


def main():
    parser = argparse.ArgumentParser(description="RGS-LaneNet demo")
    parser.add_argument("--source", default="rain_driving.mp4",
                        help="Path to video file or webcam index (e.g., 0)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device")
    parser.add_argument("--segmentation", choices=["auto", "fast"], default="auto",
                        help="Segmentation mode: auto=deep model (slower), fast=color heuristic")
    parser.add_argument("--target-lane-width-in", dest="target_lane_width_in", type=float, default=7.5,
                        help="Desired lane width in inches to enforce/measure")
    parser.add_argument("--pixels-per-inch", dest="pixels_per_inch", type=float, default=None,
                        help="Calibration scale: pixels per inch at the bottom of the frame")
    parser.add_argument("--pixels-per-foot", dest="pixels_per_foot", type=float, default=None,
                        help="Calibration scale: pixels per foot at the bottom of the frame")
    parser.add_argument("--lane-width-ft", dest="lane_width_ft", type=float, default=7.5,
                        help="Lane width to draw interior center lines (feet)")
    parser.add_argument("--width-tolerance-in", dest="width_tolerance_in", type=float, default=0.5,
                        help="Tolerance in inches before enforcing the target width")
    parser.add_argument("--save-video", dest="save_video", default=None,
                        help="Path to save processed video (e.g., output.mp4)")
    parser.add_argument("--log-csv", dest="log_csv", default=None,
                        help="Path to write per-frame measurements (CSV)")
    parser.add_argument("--no-display", action="store_true", help="Disable GUI display (headless mode)")
    parser.add_argument("--smooth-alpha", dest="smooth_alpha", type=float, default=0.7,
                        help="Exponential smoothing factor for lanes (0..1; closer to 1 = more smoothing)")
    args = parser.parse_args()

    cap = open_source(args.source)
    if not cap.isOpened():
        print(f"Failed to open source: {args.source}")
        return

    depth_est = DepthEstimator(device=args.device)

    # Prepare video writer and CSV if requested
    writer = None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    csv_file = None
    if args.save_video:
        writer = cv2.VideoWriter(args.save_video, fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    if args.log_csv:
        csv_file = open(args.log_csv, "w", encoding="utf-8")
        csv_file.write("frame,road_width_px,road_width_ft\n")

    # State for smoothing
    prev_lanes = None
    alpha = float(np.clip(args.smooth_alpha, 0.0, 1.0))

    win_name = "RGS-LaneNet"
    frame_idx = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break

        # Depth estimation (normalized 0..1)
        depth = depth_est.infer(img)

        # Semantic segmentation (road-like mask)
        seg = preprocess.semantic_segmentation(img, depth, mode=args.segmentation)

        # Lane detection (with optional target width enforcement)
        lanes = road_model.estimate_lanes(
            seg,
            depth,
            bgr_image=img,
            target_width_in=args.target_lane_width_in,
            pixels_per_inch=args.pixels_per_inch,
            width_tolerance_in=args.width_tolerance_in,
        )

        # Improved smoothing with previous frame and outlier detection
        def _x_at_y(line, y):
            x1, y1, x2, y2 = line
            if y1 == y2:
                return float(x1)
            t = (float(y) - y1) / (y2 - y1)
            return float(x1) + t * (x2 - x1)

        def _lane_distance(lane1, lane2, test_y):
            """Calculate distance between two lanes at given y position"""
            x1 = _x_at_y(lane1, test_y)
            x2 = _x_at_y(lane2, test_y)
            return abs(x1 - x2)

        def order_lr(ls):
            """Order lanes left to right by x position at bottom"""
            if len(ls) < 2:
                return ls
            h = img.shape[0]
            yb = h
            a, b = ls[0], ls[1]
            if _x_at_y(a, yb) > _x_at_y(b, yb):
                return [b, a]
            return ls

        # Apply smoothing only if we have consistent lane count and reasonable continuity
        if prev_lanes and len(prev_lanes) == len(lanes) and len(lanes) >= 1:
            h, w = img.shape[:2]
            lanes = order_lr(lanes)
            prev_lanes = order_lr(prev_lanes)
            
            # Check for outliers - reject lanes that moved too much
            smoothed = []
            for i in range(len(lanes)):
                x1, y1, x2, y2 = lanes[i]
                px1, py1, px2, py2 = prev_lanes[i]
                
                # Check if lane moved too much (outlier detection)
                max_movement = w * 0.1  # Allow 10% of image width movement
                if (_lane_distance(lanes[i], prev_lanes[i], h) < max_movement and
                    _lane_distance(lanes[i], prev_lanes[i], int(0.7*h)) < max_movement):
                    # Apply exponential smoothing
                    sx1 = int(round(alpha * px1 + (1 - alpha) * x1))
                    sx2 = int(round(alpha * px2 + (1 - alpha) * x2))
                    smoothed.append((sx1, y1, sx2, y2))
                else:
                    # Lane moved too much, use current detection without smoothing
                    smoothed.append((x1, y1, x2, y2))
            lanes = smoothed
        
        # Store current lanes for next frame
        prev_lanes = lanes

        # Ground safety confidence
        ground_conf = ground_model.compute_ground_confidence(seg, depth)

        # Turn prediction integration
        h, w = img.shape[:2]
        curvature = compute_curvature_proxy(lanes, h)
        turn_p = turn_predictor.compute_turn_probability(ground_conf, curvature)
        turn_label = "right" if curvature > 0.01 else ("left" if curvature < -0.01 else "straight")

        # Visualization
        output = visualize.draw_output(
            img,
            lanes,
            ground_conf,
            pixels_per_inch=args.pixels_per_inch,
            target_width_in=args.target_lane_width_in,
            pixels_per_foot=args.pixels_per_foot,
            lane_width_ft=args.lane_width_ft,
        )
        # Overlay warnings/info
        status = f"Safe:{ground_conf:.2f}  Turn:{turn_label} p={turn_p:.2f}"
        color = (0, 255, 0) if ground_conf > 0.5 else (0, 0, 255)
        cv2.putText(output, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        # Log measurements
        if len(lanes) >= 2 and (writer or csv_file or not args.no_display):
            yb = output.shape[0]
            dx_px = max(0.0, _x_at_y(lanes[1], yb) - _x_at_y(lanes[0], yb))
            width_ft = ""
            if args.pixels_per_foot:
                width_ft = dx_px / float(args.pixels_per_foot)
            if csv_file:
                csv_file.write(f"{frame_idx},{dx_px},{width_ft if width_ft != '' else ''}\n")

        # Write/Show
        if writer:
            writer.write(output)
        if not args.no_display:
            # cv2.imshow(win_name, output)
            # key = cv2.waitKey(1) & 0xFF
            # if key == 27:  # ESC
            #     break
            frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    if csv_file:
        csv_file.close()
    if not args.no_display:
        # cv2.destroyAllWindows()
        pass


if __name__ == "__main__":
    main()
