#!/usr/bin/env python3
"""
Debug script to visualize lane detection processing steps
"""
import cv2
import numpy as np
from models.depth_model import DepthEstimator
from utils import preprocess, road_model, visualize
import matplotlib.pyplot as plt


def debug_frame(img_path_or_frame, output_dir="debug_output"):
    """Debug a single frame and save intermediate results"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load frame
    if isinstance(img_path_or_frame, str):
        img = cv2.imread(img_path_or_frame)
    else:
        img = img_path_or_frame.copy()
    
    if img is None:
        print("Could not load image")
        return
    
    h, w = img.shape[:2]
    print(f"Image dimensions: {w}x{h}")
    
    # Initialize depth estimator
    depth_est = DepthEstimator(device="cpu")
    
    # Step 1: Original image
    cv2.imwrite(f"{output_dir}/01_original.jpg", img)
    
    # Step 2: Depth estimation
    depth = depth_est.infer(img)
    depth_vis = (depth * 255).astype(np.uint8)
    cv2.imwrite(f"{output_dir}/02_depth.jpg", depth_vis)
    
    # Step 3: Semantic segmentation
    seg = preprocess.semantic_segmentation(img, depth, mode="fast")
    seg_vis = (seg * 255).astype(np.uint8)
    cv2.imwrite(f"{output_dir}/03_segmentation.jpg", seg_vis)
    
    # Step 4: Binary lane mask (before ROI)
    binary_full = road_model._binary_lane_mask(img)
    cv2.imwrite(f"{output_dir}/04_binary_mask_full.jpg", binary_full)
    
    # Step 5: ROI mask
    roi_mask = np.zeros_like(binary_full)
    roi = np.array([[
        (int(0.1 * w), h),                    # bottom left
        (int(0.4 * w), int(0.65 * h)),        # top left
        (int(0.6 * w), int(0.65 * h)),        # top right
        (int(0.9 * w), h)                     # bottom right
    ]], dtype=np.int32)
    cv2.fillPoly(roi_mask, roi, 255)
    cv2.imwrite(f"{output_dir}/05_roi_mask.jpg", roi_mask)
    
    # Step 6: Binary mask with ROI applied
    binary_roi = cv2.bitwise_and(binary_full, roi_mask)
    cv2.imwrite(f"{output_dir}/06_binary_with_roi.jpg", binary_roi)
    
    # Step 7: Histogram analysis
    histogram = np.sum(binary_roi[int(h * 0.6):, :], axis=0)
    
    plt.figure(figsize=(12, 4))
    plt.plot(histogram)
    plt.title('Lane Detection Histogram (Bottom 40% of ROI)')
    plt.xlabel('X Position')
    plt.ylabel('Pixel Count')
    plt.axvline(x=w//2, color='r', linestyle='--', label='Midpoint')
    
    # Show detected lane bases
    midpoint = w // 2
    if np.sum(histogram[:midpoint]) > 0:
        leftx_base = int(np.argmax(histogram[:midpoint]))
        leftx_base = max(leftx_base, int(0.15 * w))
        plt.axvline(x=leftx_base, color='g', linestyle='-', label=f'Left base: {leftx_base}')
    
    if np.sum(histogram[midpoint:]) > 0:
        rightx_base = int(np.argmax(histogram[midpoint:])) + midpoint
        rightx_base = min(rightx_base, int(0.85 * w))
        plt.axvline(x=rightx_base, color='b', linestyle='-', label=f'Right base: {rightx_base}')
    
    plt.legend()
    plt.savefig(f"{output_dir}/07_histogram.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Step 8: Lane detection
    lanes = road_model.estimate_lanes(seg, depth, bgr_image=img)
    print(f"Detected {len(lanes)} lanes: {lanes}")
    
    # Step 9: Visualization
    output = visualize.draw_output(img, lanes, 0.8)  # ground_conf=0.8 for demo
    cv2.imwrite(f"{output_dir}/08_final_output.jpg", output)
    
    # Step 10: Lane overlay on binary mask for debugging
    debug_binary = cv2.cvtColor(binary_roi, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in lanes:
        cv2.line(debug_binary, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red lines
    cv2.imwrite(f"{output_dir}/09_lanes_on_binary.jpg", debug_binary)
    
    print(f"Debug images saved to {output_dir}/")
    return lanes


def debug_video_frame(video_path, frame_number=100):
    """Debug a specific frame from video"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Could not read frame {frame_number}")
        return
    
    print(f"Debugging frame {frame_number} from {video_path}")
    return debug_frame(frame, f"debug_frame_{frame_number}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        frame_num = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        debug_video_frame(video_path, frame_num)
    else:
        # Debug default video at frame 100
        debug_video_frame("rain_drive.mp4", 100)