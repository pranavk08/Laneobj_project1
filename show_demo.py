#!/usr/bin/env python3
"""
Interactive Demo Script for Lane Detection Project
Shows before/after comparisons and processing steps
"""
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from models.depth_model import DepthEstimator
from utils import preprocess, road_model, visualize

def extract_demo_frames(video_path, frame_numbers=[50, 100, 150, 200, 300]):
    """Extract specific frames from video for demo"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    for frame_num in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frames.append((frame_num, frame))
        else:
            print(f"Could not extract frame {frame_num}")
    
    cap.release()
    return frames

def create_comparison_demo(frames, output_dir="demo_comparison"):
    """Create before/after comparison for multiple frames"""
    os.makedirs(output_dir, exist_ok=True)
    
    depth_est = DepthEstimator(device="cpu")
    print("🚗 Lane Detection Demo - Processing frames...")
    
    for i, (frame_num, img) in enumerate(frames):
        print(f"  Processing frame {frame_num}...")
        
        h, w = img.shape[:2]
        
        # Step 1: Run improved lane detection
        depth = depth_est.infer(img)
        seg = preprocess.semantic_segmentation(img, depth, mode="fast")
        lanes = road_model.estimate_lanes(seg, depth, bgr_image=img)
        
        # Create visualization
        output = visualize.draw_output(img, lanes, 0.8)
        
        # Create side-by-side comparison
        comparison = np.hstack([img, output])
        
        # Add labels
        cv2.putText(comparison, "ORIGINAL", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(comparison, "LANE DETECTION", (w + 20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(comparison, f"Frame {frame_num}", (w//2 - 100, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Save comparison
        output_path = f"{output_dir}/comparison_frame_{frame_num}.jpg"
        cv2.imwrite(output_path, comparison)
        
        print(f"    Detected {len(lanes)} lanes: {lanes}")
        print(f"    Saved: {output_path}")
    
    print(f"✅ Demo images saved to {output_dir}/")

def create_processing_steps_demo(frame, frame_num, output_dir="demo_steps"):
    """Show detailed processing steps for one frame"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔍 Detailed Processing Demo - Frame {frame_num}")
    
    depth_est = DepthEstimator(device="cpu")
    h, w = frame.shape[:2]
    
    # Processing steps
    steps = []
    
    # Step 1: Original
    steps.append(("1_Original", frame.copy()))
    
    # Step 2: Depth
    depth = depth_est.infer(frame)
    depth_vis = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
    steps.append(("2_Depth", depth_vis))
    
    # Step 3: Segmentation
    seg = preprocess.semantic_segmentation(frame, depth, mode="fast")
    seg_vis = cv2.cvtColor((seg * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    steps.append(("3_Segmentation", seg_vis))
    
    # Step 4: Binary mask
    binary = road_model._binary_lane_mask(frame)
    binary_vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    steps.append(("4_Binary_Mask", binary_vis))
    
    # Step 5: ROI
    roi_mask = np.zeros_like(binary)
    roi = np.array([[
        (int(0.1 * w), h), (int(0.4 * w), int(0.65 * h)),
        (int(0.6 * w), int(0.65 * h)), (int(0.9 * w), h)
    ]], dtype=np.int32)
    cv2.fillPoly(roi_mask, roi, 255)
    binary_roi = cv2.bitwise_and(binary, roi_mask)
    roi_vis = cv2.cvtColor(binary_roi, cv2.COLOR_GRAY2BGR)
    # Draw ROI outline
    cv2.polylines(roi_vis, roi, True, (0, 255, 0), 3)
    steps.append(("5_ROI_Applied", roi_vis))
    
    # Step 6: Final result
    lanes = road_model.estimate_lanes(seg, depth, bgr_image=frame)
    final = visualize.draw_output(frame, lanes, 0.8)
    steps.append(("6_Final_Result", final))
    
    # Create grid layout (2x3)
    step_images = []
    for name, img in steps:
        # Resize for grid display
        img_resized = cv2.resize(img, (w//2, h//2))
        # Add title
        cv2.putText(img_resized, name.replace('_', ' '), (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        step_images.append(img_resized)
        
        # Save individual step
        cv2.imwrite(f"{output_dir}/{name}.jpg", img)
    
    # Create grid
    top_row = np.hstack(step_images[:3])
    bottom_row = np.hstack(step_images[3:])
    grid = np.vstack([top_row, bottom_row])
    
    cv2.imwrite(f"{output_dir}/processing_steps_grid.jpg", grid)
    
    print(f"    Detected {len(lanes)} lanes")
    print(f"✅ Processing steps saved to {output_dir}/")

def show_performance_stats(video_path):
    """Show performance statistics"""
    print("\n📊 Performance Statistics:")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    cap.release()
    
    print(f"  📹 Video: {os.path.basename(video_path)}")
    print(f"  ⏱️  Duration: {duration:.1f} seconds")
    print(f"  🎬 Total Frames: {total_frames}")
    print(f"  📊 Original FPS: {fps:.1f}")
    print(f"  🖥️  Processing: ~4-5 FPS on CPU")
    print(f"  ⚡ Estimated Processing Time: {total_frames/4.5:.1f} seconds")

def main():
    print("🎬 LANE DETECTION PROJECT DEMO")
    print("=" * 50)
    
    video_path = "project_video.mp4"
    
    # Show performance stats
    show_performance_stats(video_path)
    
    print(f"\n🎯 Extracting demo frames from {video_path}...")
    frames = extract_demo_frames(video_path, [50, 100, 150, 200, 300])
    
    print(f"✅ Extracted {len(frames)} frames")
    
    # Create comparison demo
    print(f"\n🔄 Creating before/after comparison...")
    create_comparison_demo(frames)
    
    # Detailed processing for one frame
    print(f"\n🔍 Creating detailed processing demo...")
    if frames:
        frame_num, frame = frames[2]  # Use middle frame
        create_processing_steps_demo(frame, frame_num)
    
    # Show file outputs
    print(f"\n📁 Generated Demo Files:")
    print(f"  📂 demo_comparison/ - Before/after comparisons")
    print(f"  📂 demo_steps/ - Detailed processing steps")
    
    # Show recent video outputs
    print(f"\n🎥 Recent Video Outputs:")
    video_files = [f for f in os.listdir('.') if f.endswith('.mp4') and 'demo' in f]
    for video in sorted(video_files, reverse=True)[:3]:
        size_mb = os.path.getsize(video) / (1024*1024)
        print(f"  🎬 {video} ({size_mb:.1f} MB)")
    
    print(f"\n🎉 DEMO COMPLETE! Check the generated images and videos.")
    print(f"💡 Tip: Open the comparison images to see the lane detection results!")

if __name__ == "__main__":
    main()