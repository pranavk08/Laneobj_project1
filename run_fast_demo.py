import argparse
import cv2
import numpy as np

from models.depth_model import DepthEstimator
from models import turn_predictor
from utils import preprocess, road_model, ground_model, visualize


def main():
    parser = argparse.ArgumentParser(description="RGS-LaneNet fast demo")
    parser.add_argument("--source", default="test_short.mp4", help="Path to video file")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale factor to reduce resolution (0.5 = half size)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Failed to open source: {args.source}")
        return

    depth_est = DepthEstimator(device=args.device)
    
    frame_idx = 0
    fps_start = cv2.getTickCount()
    
    print(f"Processing with scale={args.scale}, device={args.device}")
    print("Press 'q' to quit")
    
    while True:
        ret, img = cap.read()
        if not ret:
            break
        
        # Resize for faster processing
        if args.scale != 1.0:
            img = cv2.resize(img, None, fx=args.scale, fy=args.scale)
        
        h, w = img.shape[:2]
        
        # Depth estimation
        depth = depth_est.infer(img)
        
        # Fast segmentation only
        seg = preprocess.semantic_segmentation(img, depth, mode="fast")
        
        # Lane detection
        lanes = road_model.estimate_lanes(seg, depth, bgr_image=img)
        
        # Ground confidence
        ground_conf = ground_model.compute_ground_confidence(seg, depth)
        
        # Visualization
        output = visualize.draw_output(img, lanes, ground_conf)
        
        # FPS calculation
        frame_idx += 1
        if frame_idx % 30 == 0:
            fps_elapsed = (cv2.getTickCount() - fps_start) / cv2.getTickFrequency()
            fps = 30 / fps_elapsed if fps_elapsed > 0 else 0
            fps_start = cv2.getTickCount()
            print(f"Frame {frame_idx}, FPS: {fps:.1f}, Ground conf: {ground_conf:.2f}")
        
        # Show
        cv2.imshow("Fast Demo", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_idx} frames")


if __name__ == "__main__":
    main()
