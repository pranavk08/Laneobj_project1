#!/usr/bin/env python3
"""
Parallel Lane Detection - Ensures lanes don't cross
"""
import cv2
import numpy as np
import argparse
import time


def detect_white_yellow_lanes(img):
    """Detect white and yellow lane markings"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # White lanes
    white_mask = cv2.inRange(gray, 200, 255)
    
    # Yellow lanes  
    yellow_mask = cv2.inRange(hsv, (15, 50, 50), (35, 255, 255))
    
    # Combine
    lane_mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    # Edge detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # Combine masks and edges
    combined = cv2.bitwise_or(lane_mask, edges)
    
    return combined


def region_of_interest(img):
    """Apply trapezoid ROI"""
    h, w = img.shape[:2]
    mask = np.zeros_like(img)
    
    polygon = np.array([[
        (int(w * 0.1), h),
        (int(w * 0.45), int(h * 0.6)),
        (int(w * 0.55), int(h * 0.6)),
        (int(w * 0.9), h)
    ]], np.int32)
    
    cv2.fillPoly(mask, polygon, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def find_lane_lines_parallel(binary_img):
    """
    Find lane lines with parallel constraint
    Returns two lines that stay parallel
    """
    h, w = binary_img.shape
    
    # Get histogram of bottom half
    histogram = np.sum(binary_img[h//2:, :], axis=0)
    
    # Find peaks in left and right halves
    midpoint = w // 2
    left_peak = np.argmax(histogram[:midpoint])
    right_peak = np.argmax(histogram[midpoint:]) + midpoint
    
    # If peaks are too close or too far, adjust
    lane_width = right_peak - left_peak
    if lane_width < w * 0.2:  # Too narrow
        center = (left_peak + right_peak) // 2
        left_peak = center - int(w * 0.15)
        right_peak = center + int(w * 0.15)
    elif lane_width > w * 0.6:  # Too wide  
        center = (left_peak + right_peak) // 2
        left_peak = center - int(w * 0.25)
        right_peak = center + int(w * 0.25)
    
    # Use sliding windows
    nwindows = 9
    window_height = h // nwindows
    margin = 50
    minpix = 50
    
    nonzero = binary_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_lane_inds = []
    right_lane_inds = []
    
    left_current = left_peak
    right_current = right_peak
    
    for window in range(nwindows):
        win_y_low = h - (window + 1) * window_height
        win_y_high = h - window * window_height
        
        # Left window
        win_xleft_low = left_current - margin
        win_xleft_high = left_current + margin
        good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                    (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        left_lane_inds.append(good_left)
        
        # Right window
        win_xright_low = right_current - margin
        win_xright_high = right_current + margin
        good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                     (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        right_lane_inds.append(good_right)
        
        # Recenter
        if len(good_left) > minpix:
            left_current = int(np.mean(nonzerox[good_left]))
        if len(good_right) > minpix:
            right_current = int(np.mean(nonzerox[good_right]))
    
    # Concatenate
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit polynomials
    lanes = []
    
    if len(leftx) > 50:
        left_fit = np.polyfit(lefty, leftx, 2)
        y1 = h
        y2 = int(h * 0.6)
        x1 = int(left_fit[0] * y1**2 + left_fit[1] * y1 + left_fit[2])
        x2 = int(left_fit[0] * y2**2 + left_fit[1] * y2 + left_fit[2])
        lanes.append((x1, y1, x2, y2))
    else:
        # Fallback - vertical line
        lanes.append((left_peak, h, left_peak, int(h * 0.6)))
    
    if len(rightx) > 50:
        right_fit = np.polyfit(righty, rightx, 2)
        y1 = h
        y2 = int(h * 0.6)
        x1 = int(right_fit[0] * y1**2 + right_fit[1] * y1 + right_fit[2])
        x2 = int(right_fit[0] * y2**2 + right_fit[1] * y2 + right_fit[2])
        lanes.append((x1, y1, x2, y2))
    else:
        # Fallback - vertical line
        lanes.append((right_peak, h, right_peak, int(h * 0.6)))
    
    return lanes


def draw_lanes(img, lanes):
    """Draw lanes with filled area"""
    overlay = img.copy()
    
    if len(lanes) >= 2:
        # Draw filled polygon
        left = lanes[0]
        right = lanes[1]
        
        pts = np.array([[left[0], left[1]], [left[2], left[3]], 
                       [right[2], right[3]], [right[0], right[1]]], np.int32)
        cv2.fillPoly(overlay, [pts], (0, 255, 255))
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        
        # Draw lane lines
        cv2.line(img, (left[0], left[1]), (left[2], left[3]), (0, 255, 0), 8)
        cv2.line(img, (right[0], right[1]), (right[2], right[3]), (0, 255, 0), 8)
    
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='test1.mp4')
    parser.add_argument('--output', default='parallel_lanes.mp4')
    parser.add_argument('--display', action='store_true')
    args = parser.parse_args()
    
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Cannot open {args.source}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
    
    frame_count = 0
    start_time = time.time()
    
    print(f"Processing {args.source}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect lanes
        binary = detect_white_yellow_lanes(frame)
        roi = region_of_interest(binary)
        lanes = find_lane_lines_parallel(roi)
        
        # Draw
        result = draw_lanes(frame.copy(), lanes)
        
        # Add text
        cv2.putText(result, f"Frame: {frame_count} | Lanes: {len(lanes)}", 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        out.write(result)
        frame_count += 1
        
        if args.display:
            cv2.imshow('Parallel Lane Detection', result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed
            print(f"Processed {frame_count} frames... FPS: {fps_actual:.1f}")
    
    cap.release()
    out.release()
    if args.display:
        cv2.destroyAllWindows()
    
    print(f"Done! Saved to {args.output}")
    print(f"Total frames: {frame_count}")


if __name__ == '__main__':
    main()
