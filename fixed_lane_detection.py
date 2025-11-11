#!/usr/bin/env python3
"""
Fixed lane detection that properly detects driving lane boundaries
"""
import cv2
import numpy as np
from models.depth_model import DepthEstimator
from utils import preprocess, visualize
import argparse


def precise_lane_mask(bgr: np.ndarray) -> np.ndarray:
    """Enhanced binary mask specifically for lane boundaries"""
    h, w = bgr.shape[:2]
    
    # Convert to multiple color spaces
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    
    # Focus primarily on WHITE lane markings (driving lane boundaries)
    # Use stricter thresholds to avoid false positives
    white_binary = np.zeros_like(gray)
    white_binary[(gray > 190)] = 1  # Higher threshold for cleaner white detection
    
    # HLS white detection - more selective
    hls_white = np.zeros_like(gray)
    hls_white[((hls[:,:,1] > 190) & (hls[:,:,2] > 100))] = 1
    
    # Combine white detections
    white_mask = np.zeros_like(gray)
    white_mask[(white_binary == 1) | (hls_white == 1)] = 1
    
    # Enhanced edge detection for lane markings
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Sobel X (vertical edges) - primary for lane lines
    sobelx = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=7)
    abs_sobelx = np.absolute(sobelx)
    
    # Apply directional masking to focus on vertical lines
    sobely = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=7)
    abs_sobely = np.absolute(sobely)
    
    # Calculate gradient direction
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    
    # Focus on near-vertical lines (lane markings)
    dir_binary = np.zeros_like(gray)
    dir_binary[(grad_dir > 1.2) | (grad_dir < 0.4)] = 1  # Vertical-ish lines
    
    # Scale and threshold Sobel X
    if np.max(abs_sobelx) > 0:
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        grad_binary = np.zeros_like(scaled_sobel)
        grad_binary[(scaled_sobel >= 30) & (scaled_sobel <= 255)] = 1  # Higher threshold
    else:
        grad_binary = np.zeros_like(gray)
    
    # Combine detections with emphasis on white + vertical edges
    combined_binary = np.zeros_like(gray)
    combined_binary[(white_mask == 1) | 
                   ((grad_binary == 1) & (dir_binary == 1))] = 1
    
    # Clean up with morphological operations
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(combined_binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    return cleaned * 255


def detect_driving_lanes(bgr_image: np.ndarray) -> list:
    """Detect the actual driving lane boundaries (not center dividers)"""
    h, w = bgr_image.shape[:2]
    
    # Get binary mask focused on lane markings
    binary = precise_lane_mask(bgr_image)
    
    # Apply ROI - focus on driving area
    roi_mask = np.zeros_like(binary)
    roi = np.array([[
        (int(0.05 * w), h),                    # bottom left
        (int(0.43 * w), int(0.6 * h)),         # top left
        (int(0.57 * w), int(0.6 * h)),         # top right  
        (int(0.95 * w), h)                     # bottom right
    ]], dtype=np.int32)
    cv2.fillPoly(roi_mask, roi, 255)
    binary = cv2.bitwise_and(binary, roi_mask)
    
    # Enhanced histogram analysis with peak filtering
    bottom_section = binary[int(h * 0.65):, :]
    histogram = np.sum(bottom_section, axis=0)
    
    # Smooth histogram to reduce noise
    from scipy.ndimage import gaussian_filter1d
    histogram = gaussian_filter1d(histogram.astype(float), sigma=3)
    
    # Find peaks more intelligently
    midpoint = w // 2
    
    # For driving lanes, we want:
    # 1. Left boundary (left white line of our lane)
    # 2. Right boundary (right white line of our lane)
    # NOT the center divider
    
    # Search for left driving lane boundary (should be to the left of center)
    left_search_area = histogram[int(0.1*w):int(0.48*w)]  # Search left area, avoid center
    if np.max(left_search_area) > 100:  # Higher threshold for confidence
        # Find the strongest peak in left area
        peaks = []
        for i in range(20, len(left_search_area) - 20):
            if (left_search_area[i] > left_search_area[i-20:i].max() and 
                left_search_area[i] > left_search_area[i+1:i+21].max() and
                left_search_area[i] > 100):
                peaks.append((left_search_area[i], i + int(0.1*w)))
        
        if peaks:
            leftx_base = max(peaks)[1]  # Strongest peak
        else:
            leftx_base = np.argmax(left_search_area) + int(0.1*w)
    else:
        leftx_base = int(0.25 * w)  # Default position
    
    # Search for right driving lane boundary (should be to the right of center)
    right_search_area = histogram[int(0.52*w):int(0.9*w)]  # Search right area, avoid center
    if np.max(right_search_area) > 100:  # Higher threshold for confidence
        peaks = []
        for i in range(20, len(right_search_area) - 20):
            if (right_search_area[i] > right_search_area[i-20:i].max() and 
                right_search_area[i] > right_search_area[i+1:i+21].max() and
                right_search_area[i] > 100):
                peaks.append((right_search_area[i], i + int(0.52*w)))
        
        if peaks:
            rightx_base = max(peaks)[1]  # Strongest peak
        else:
            rightx_base = np.argmax(right_search_area) + int(0.52*w)
    else:
        rightx_base = int(0.75 * w)  # Default position
    
    # Ensure reasonable lane width (not too narrow, not too wide)
    lane_width = rightx_base - leftx_base
    if lane_width < w * 0.15:  # Too narrow
        center = (leftx_base + rightx_base) // 2
        leftx_base = center - int(w * 0.1)
        rightx_base = center + int(w * 0.1)
    elif lane_width > w * 0.6:  # Too wide
        center = (leftx_base + rightx_base) // 2
        leftx_base = center - int(w * 0.15)
        rightx_base = center + int(w * 0.15)
    
    # Sliding window search with better parameters
    nwindows = 10
    window_height = int(h / nwindows)
    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    margin = max(int(w * 0.04), 50)  # Smaller margin for precision
    minpix = 25  # Lower requirement
    
    left_lane_inds = []
    right_lane_inds = []
    
    for window in range(nwindows):
        win_y_low = h - (window + 1) * window_height
        win_y_high = h - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                         (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # Recenter with validation
        if len(good_left_inds) > minpix:
            new_leftx = int(np.mean(nonzerox[good_left_inds]))
            # Ensure it doesn't drift too far from expected position
            if abs(new_leftx - leftx_current) < w * 0.1:  # Max 10% drift
                leftx_current = new_leftx
        
        if len(good_right_inds) > minpix:
            new_rightx = int(np.mean(nonzerox[good_right_inds]))
            if abs(new_rightx - rightx_current) < w * 0.1:
                rightx_current = new_rightx
    
    # Extract and fit polynomials
    left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([], dtype=int)
    right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([], dtype=int)
    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    def fit_driving_lane(xs, ys):
        """Fit polynomial for driving lane boundary"""
        if len(xs) < 20:  # Need enough points
            return None
        
        try:
            # Weighted fit - more weight to bottom of image
            weights = np.exp(-0.002 * ys)  # Exponential weighting
            fit = np.polyfit(ys, xs, 2, w=weights)
            
            y_bottom = h
            y_top = int(h * 0.5)  # Extend higher up
            
            x_bottom = int(fit[0] * y_bottom**2 + fit[1] * y_bottom + fit[2])
            x_top = int(fit[0] * y_top**2 + fit[1] * y_top + fit[2])
            
            # Validate lane position
            if (0 <= x_bottom < w and 0 <= x_top < w and 
                abs(x_bottom - x_top) < w * 0.3):  # Reasonable curvature
                return (x_bottom, y_bottom, x_top, y_top)
        except:
            pass
        return None
    
    # Fit lanes
    left_line = fit_driving_lane(leftx, lefty)
    right_line = fit_driving_lane(rightx, righty)
    
    lanes = []
    if left_line:
        lanes.append(left_line)
    if right_line:
        lanes.append(right_line)
    
    # If we don't have both lanes, create synthetic ones based on typical lane width
    if len(lanes) == 1:
        detected = lanes[0]
        x_bottom, y_bottom, x_top, y_top = detected
        typical_width = int(w * 0.22)  # Typical driving lane width
        
        if x_bottom < w // 2:  # It's a left lane
            synthetic_right = (x_bottom + typical_width, y_bottom, 
                             x_top + typical_width, y_top)
            if synthetic_right[0] < w * 0.9 and synthetic_right[2] < w * 0.9:
                lanes.append(synthetic_right)
        else:  # It's a right lane
            synthetic_left = (x_bottom - typical_width, y_bottom,
                            x_top - typical_width, y_top)
            if synthetic_left[0] > w * 0.1 and synthetic_left[2] > w * 0.1:
                lanes.insert(0, synthetic_left)
    
    elif len(lanes) == 0:
        # Fallback lanes based on detected positions
        left_fallback = (leftx_base, h, leftx_base, int(h * 0.5))
        right_fallback = (rightx_base, h, rightx_base, int(h * 0.5))
        lanes = [left_fallback, right_fallback]
    
    # Ensure proper ordering
    if len(lanes) >= 2:
        lanes.sort(key=lambda lane: lane[0])
    
    return lanes


def main():
    parser = argparse.ArgumentParser(description="Fixed Lane Detection for Driving Lanes")
    parser.add_argument("--source", default="DUAL_LANE_DEMO.mp4", help="Video source")
    parser.add_argument("--output", default="fixed_lanes_output.mp4", help="Output video")
    parser.add_argument("--fps-reduce", type=int, default=1, help="Frame rate reduction factor")
    args = parser.parse_args()
    
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Failed to open {args.source}")
        return
    
    depth_est = DepthEstimator(device="cpu")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    output_fps = fps / args.fps_reduce
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, output_fps,
                           (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    frame_count = 0
    processed_frames = 0
    
    while True:
        ret, img = cap.read()
        if not ret:
            break
        
        # Skip frames for FPS reduction
        if frame_count % args.fps_reduce != 0:
            frame_count += 1
            continue
        
        # Use fixed lane detection
        lanes = detect_driving_lanes(img)
        
        # Simple visualization (replace the complex visualize function)
        output = img.copy()
        
        # Draw lanes
        for i, (x1, y1, x2, y2) in enumerate(lanes):
            color = (0, 255, 0)  # Green for lanes
            cv2.line(output, (x1, y1), (x2, y2), color, 8)
            
            # Add lane label
            label = f"Lane {i+1}"
            cv2.putText(output, label, (x2 - 30, y2 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add info
        info = f"Frame: {processed_frames} | Detected Lanes: {len(lanes)}"
        cv2.putText(output, info, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(output, "Fixed Lane Detection - Driving Boundaries Only", 
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        writer.write(output)
        processed_frames += 1
        frame_count += 1
        
        if processed_frames % 50 == 0:
            print(f"Processed {processed_frames} frames...")
    
    cap.release()
    writer.release()
    print(f"Fixed lane detection saved to {args.output}")
    print(f"Processed {processed_frames} frames from {frame_count} total frames")


if __name__ == "__main__":
    main()