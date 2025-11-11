import argparse
import cv2
import numpy as np

from models.depth_model import DepthEstimator
from models import turn_predictor
from utils import preprocess, road_model, ground_model, visualize


def main():
    parser = argparse.ArgumentParser(description="RGS-LaneNet quality demo")
    parser.add_argument("--source", default="test_short.mp4", help="Path to video file")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device")
    parser.add_argument("--scale", type=float, default=0.75, help="Scale factor (0.75 = good balance)")
    parser.add_argument("--save", default=None, help="Save output video")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Failed to open source: {args.source}")
        return

    depth_est = DepthEstimator(device=args.device)
    
    # Video writer setup
    writer = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * args.scale)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * args.scale)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))
    
    frame_idx = 0
    fps_start = cv2.getTickCount()
    
    print(f"Processing with scale={args.scale}, device={args.device}")
    print("Using AUTO segmentation (deep learning) for better accuracy")
    print("Press 'q' to quit")
    
    while True:
        ret, img = cap.read()
        if not ret:
            break
        
        # Resize for processing
        if args.scale != 1.0:
            img = cv2.resize(img, None, fx=args.scale, fy=args.scale)
        
        h, w = img.shape[:2]
        
        # Depth estimation
        depth = depth_est.infer(img)
        
        # Use AUTO mode for better segmentation (deep learning)
        seg = preprocess.semantic_segmentation(img, depth, mode="auto")
        
        # Lane detection with better parameters
        lanes = road_model.estimate_lanes(seg, depth, bgr_image=img)
        
        # Ground confidence
        ground_conf = ground_model.compute_ground_confidence(seg, depth)
        
        # Visualization
        output = visualize.draw_output(img, lanes, ground_conf)
        
        # Add info overlay
        cv2.putText(output, f"Frame: {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(output, f"Lanes: {len(lanes)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(output, f"Ground: {ground_conf:.2f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # FPS calculation
        frame_idx += 1
        if frame_idx % 30 == 0:
            fps_elapsed = (cv2.getTickCount() - fps_start) / cv2.getTickFrequency()
            fps = 30 / fps_elapsed if fps_elapsed > 0 else 0
            fps_start = cv2.getTickCount()
            print(f"Frame {frame_idx}, FPS: {fps:.1f}, Lanes: {len(lanes)}, Ground: {ground_conf:.2f}")
        
        # Save/Show
        if writer:
            writer.write(output)
        
        cv2.imshow("Quality Demo", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_idx} frames")
    if args.save:
        print(f"Saved to: {args.save}")


if __name__ == "__main__":
    main()
