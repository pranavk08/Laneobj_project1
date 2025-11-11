#!/usr/bin/env python3
"""
Improved lane detection with better parameter tuning and filtering
"""
import cv2
import numpy as np
from models.depth_model import DepthEstimator
from utils import preprocess, road_model, visualize
import argparse


def improved_binary_mask(bgr: np.ndarray) -> np.ndarray:
    """Enhanced binary mask for better lane detection"""
    h, w = bgr.shape[:2]
    
    # Convert to multiple color spaces
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # Enhanced white lane detection
    white_binary = np.zeros_like(gray)
    # Lower threshold for better detection in shadows
    white_binary[(gray > 180)] = 1
    
    # HLS white detection with relaxed thresholds
    hls_white = np.zeros_like(gray)
    hls_white[((hls[:,:,1] > 180) & (hls[:,:,2] > 80))] = 1
    
    # Enhanced yellow detection
    yellow_hsv = np.zeros_like(gray)
    yellow_hsv[((hsv[:,:,0] >= 15) & (hsv[:,:,0] <= 35) & 
                (hsv[:,:,1] >= 50) & (hsv[:,:,2] >= 50))] = 1
    
    # Yellow in HLS 
    yellow_hls = np.zeros_like(gray)
    yellow_hls[((hls[:,:,0] >= 15) & (hls[:,:,0] <= 35) & 
                (hls[:,:,1] >= 20) & (hls[:,:,2] >= 50))] = 1
    
    # Combine all color detections
    color_binary = np.zeros_like(gray)
    color_binary[(white_binary == 1) | (hls_white == 1) | 
                 (yellow_hsv == 1) | (yellow_hls == 1)] = 1
    
    # Enhanced gradient detection
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Sobel X with better parameters
    sobelx = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=7)
    abs_sobelx = np.absolute(sobelx)
    
    # Sobel Y for better edge detection
    sobely = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=7)
    abs_sobely = np.absolute(sobely)
    
    # Gradient magnitude and direction
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    
    # Scale gradients
    if np.max(grad_mag) > 0:
        scaled_sobel = np.uint8(255 * grad_mag / np.max(grad_mag))
        grad_binary = np.zeros_like(scaled_sobel)
        # Lower threshold for better detection
        grad_binary[(scaled_sobel >= 20) & (scaled_sobel <= 255)] = 1
    else:
        grad_binary = np.zeros_like(gray)
    
    # Direction filtering (vertical lines)
    dir_binary = np.zeros_like(gray)
    dir_binary[(grad_dir > 0.7) & (grad_dir < 1.3)] = 1
    
    # Combine all detections
    combined_binary = np.zeros_like(gray)
    combined_binary[(color_binary == 1) | 
                   ((grad_binary == 1) & (dir_binary == 1))] = 1
    
    # Morphological operations with optimized kernels
    kernel_close = np.ones((5,5), np.uint8)
    kernel_open = np.ones((3,3), np.uint8)
    
    cleaned = cv2.morphologyEx(combined_binary.astype(np.uint8), 
                              cv2.MORPH_CLOSE, kernel_close)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)
    
    return cleaned * 255


def improved_lane_estimation(seg, depth, bgr_image, **kwargs):
    """Improved lane estimation with better parameters"""
    h, w = bgr_image.shape[:2]
    
    # Use improved binary mask
    binary = improved_binary_mask(bgr_image)
    
    # Enhanced ROI - more conservative trapezoid
    roi_mask = np.zeros_like(binary)
    roi = np.array([[
        (int(0.05 * w), h),                    # bottom left - closer to edge
        (int(0.42 * w), int(0.62 * h)),        # top left - slightly adjusted
        (int(0.58 * w), int(0.62 * h)),        # top right
        (int(0.95 * w), h)                     # bottom right - closer to edge
    ]], dtype=np.int32)
    cv2.fillPoly(roi_mask, roi, 255)
    binary = cv2.bitwise_and(binary, roi_mask)
    
    # Improved histogram analysis - use bottom 30% instead of 25%
    bottom_section = binary[int(h * 0.7):, :]
    histogram = np.sum(bottom_section, axis=0)
    
    # Apply smoothing to histogram to reduce noise
    from scipy.ndimage import gaussian_filter1d
    histogram = gaussian_filter1d(histogram.astype(float), sigma=2)
    
    # Find peaks with better parameters
    midpoint = w // 2
    
    # Left lane base with improved peak detection
    left_half = histogram[:midpoint]
    if np.max(left_half) > 30:  # Lower threshold
        # Find multiple peaks and choose the strongest one
        peaks = []
        for i in range(10, len(left_half) - 10):
            if (left_half[i] > left_half[i-10:i].max() and 
                left_half[i] > left_half[i+1:i+11].max() and
                left_half[i] > 30):
                peaks.append((left_half[i], i))
        
        if peaks:
            leftx_base = max(peaks)[1]  # Choose strongest peak
        else:
            leftx_base = np.argmax(left_half)
            
        # Ensure reasonable bounds
        leftx_base = max(leftx_base, int(0.1 * w))
        leftx_base = min(leftx_base, int(0.45 * w))
    else:
        leftx_base = w // 4
    
    # Similar for right lane
    right_half = histogram[midpoint:]
    if np.max(right_half) > 30:
        peaks = []
        for i in range(10, len(right_half) - 10):
            if (right_half[i] > right_half[i-10:i].max() and 
                right_half[i] > right_half[i+1:i+11].max() and
                right_half[i] > 30):
                peaks.append((right_half[i], i + midpoint))
        
        if peaks:
            rightx_base = max(peaks)[1]
        else:
            rightx_base = np.argmax(right_half) + midpoint
            
        # Ensure reasonable bounds
        rightx_base = max(rightx_base, int(0.55 * w))
        rightx_base = min(rightx_base, int(0.9 * w))
    else:
        rightx_base = w * 3 // 4
    
    # Enhanced sliding window parameters
    nwindows = 12  # More windows for better tracking
    window_height = int(h / nwindows)
    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Adaptive margin based on image size
    margin = int(w * 0.06)  # 6% of image width
    minpix = 40  # Lower minimum pixels
    
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
        
        # Recenter with smoothing
        if len(good_left_inds) > minpix:
            new_leftx = int(np.mean(nonzerox[good_left_inds]))
            # Smooth the transition
            leftx_current = int(0.7 * leftx_current + 0.3 * new_leftx)
        
        if len(good_right_inds) > minpix:
            new_rightx = int(np.mean(nonzerox[good_right_inds]))
            rightx_current = int(0.7 * rightx_current + 0.3 * new_rightx)
    
    # Concatenate and fit polynomials
    left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([], dtype=int)
    right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([], dtype=int)
    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    lanes = []
    
    def improved_fit_line(xs, ys):
        """Improved line fitting with validation"""
        if len(xs) < 30:  # Lower threshold
            return None
        
        try:
            # Use weighted fitting - give more weight to bottom pixels
            weights = np.exp(-0.001 * ys)  # Bottom pixels get higher weight
            fit = np.polyfit(ys, xs, 2, w=weights)
            
            y_bottom = h
            y_top = int(h * 0.55)  # Extend higher
            
            x_bottom = int(fit[0] * y_bottom**2 + fit[1] * y_bottom + fit[2])
            x_top = int(fit[0] * y_top**2 + fit[1] * y_top + fit[2])
            
            # Enhanced validation
            if (0 <= x_bottom < w and 0 <= x_top < w and 
                abs(x_bottom - x_top) < w * 0.4):  # Reasonable slope
                return (x_bottom, y_bottom, x_top, y_top)
            
        except Exception:
            pass
        return None
    
    # Fit lanes
    left_line = improved_fit_line(leftx, lefty)
    right_line = improved_fit_line(rightx, righty)
    
    # Add detected lanes
    if left_line:
        lanes.append(left_line)
    if right_line:
        lanes.append(right_line)
    
    # Fallback if no lanes detected
    if len(lanes) == 0:
        left_fallback = (leftx_base, h, leftx_base, int(h * 0.55))
        right_fallback = (rightx_base, h, rightx_base, int(h * 0.55))
        lanes = [left_fallback, right_fallback]
    
    # Synthesize missing lane
    elif len(lanes) == 1:
        detected = lanes[0]
        x_bottom, y_bottom, x_top, y_top = detected
        
        # Determine typical lane width (adaptive)
        lane_width = max(int(w * 0.2), 150)  # At least 150 pixels
        
        if x_bottom < w // 2:  # Left lane detected
            synthetic_right = (x_bottom + lane_width, y_bottom, 
                             x_top + lane_width, y_top)
            if synthetic_right[0] < w and synthetic_right[2] < w:
                lanes.append(synthetic_right)
        else:  # Right lane detected
            synthetic_left = (x_bottom - lane_width, y_bottom,
                            x_top - lane_width, y_top)
            if synthetic_left[0] >= 0 and synthetic_left[2] >= 0:
                lanes.insert(0, synthetic_left)
    
    # Sort lanes left to right
    if len(lanes) >= 2:
        lanes.sort(key=lambda lane: lane[0])
    
    return lanes


def main():
    parser = argparse.ArgumentParser(description="Improved Lane Detection Demo")
    parser.add_argument("--source", default="DUAL_LANE_DEMO.mp4", 
                       help="Video source")
    parser.add_argument("--output", default="improved_lane_output.mp4",
                       help="Output video path")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--display", action="store_true",
                       help="Show live display window")
    args = parser.parse_args()
    
    # Monkey patch the improved function
    road_model.estimate_lanes = improved_lane_estimation
    
    # Run demo with improved detection
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Failed to open {args.source}")
        return
    
    depth_est = DepthEstimator(device=args.device)
    print(f"Using device: {args.device}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, 
                           (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    frame_count = 0
    import time
    start_time = time.time()
    fps_counter = 0
    fps_display = 0
    
    print("Processing... Press 'q' to quit (if display is enabled)")
    
    while True:
        ret, img = cap.read()
        if not ret:
            break
        
        depth = depth_est.infer(img)
        seg = preprocess.semantic_segmentation(img, depth, mode="fast")
        
        # Use improved lane detection
        lanes = improved_lane_estimation(seg, depth, img)
        
        # Visualize
        output = visualize.draw_output(img, lanes, 0.8)
        
        # Calculate FPS
        fps_counter += 1
        if fps_counter >= 30:
            elapsed = time.time() - start_time
            fps_display = fps_counter / elapsed
            fps_counter = 0
            start_time = time.time()
        
        # Add frame info with FPS
        cv2.putText(output, f"Frame: {frame_count} | Lanes: {len(lanes)} | FPS: {fps_display:.1f}", 
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        writer.write(output)
        frame_count += 1
        
        # Display window if requested
        if args.display:
            cv2.imshow("Improved Lane Detection", output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames... FPS: {fps_display:.1f}")
    
    cap.release()
    writer.release()
    if args.display:
        cv2.destroyAllWindows()
    print(f"Improved detection saved to {args.output}")
    print(f"Total frames processed: {frame_count}")


if __name__ == "__main__":
    main()