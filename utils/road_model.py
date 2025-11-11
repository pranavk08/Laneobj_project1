import numpy as np
import cv2
from typing import List, Optional, Tuple

Line = Tuple[int, int, int, int]


def _synthesize_left_lane(right_line: Line, width: int, height: int) -> Line:
    """Synthesize left lane based on right lane with typical lane width"""
    rx1, ry1, rx2, ry2 = right_line
    # Typical highway lane width is about 25% of image width
    lane_width = int(0.25 * width)
    lx1 = max(rx1 - lane_width, int(0.1 * width))
    lx2 = max(rx2 - lane_width, int(0.1 * width))
    return (lx1, ry1, lx2, ry2)


def _synthesize_right_lane(left_line: Line, width: int, height: int) -> Line:
    """Synthesize right lane based on left lane with typical lane width"""
    lx1, ly1, lx2, ly2 = left_line
    # Typical highway lane width is about 25% of image width
    lane_width = int(0.25 * width)
    rx1 = min(lx1 + lane_width, int(0.9 * width))
    rx2 = min(lx2 + lane_width, int(0.9 * width))
    return (rx1, ly1, rx2, ly2)


def _auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(gray, lower, upper, L2gradient=True)
    return edges


def _binary_lane_mask(bgr: np.ndarray) -> np.ndarray:
    """Completely redesigned binary mask for accurate lane line detection"""
    h, w = bgr.shape[:2]
    
    # Step 1: Precise white lane detection
    # Convert to multiple color spaces
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # Focus on WHITE lane boundaries (driving lanes) - avoid center dividers
    white_binary = np.zeros_like(gray)
    white_binary[(gray > 190)] = 1  # Higher threshold to focus on clear white lines
    
    # HLS white detection - more selective for driving lanes
    hls_white = np.zeros_like(gray)
    hls_white[((hls[:,:,1] > 190) & (hls[:,:,2] > 100))] = 1
    
    # Combine white detections
    white_mask = np.zeros_like(gray)
    white_mask[(white_binary == 1) | (hls_white == 1)] = 1
    
    # Step 2: Minimize yellow detection to avoid center dividers
    # Only detect very clear yellow markings (reduce false positives)
    yellow_binary = np.zeros_like(gray)
    # More restrictive yellow detection to avoid center line interference
    yellow_binary[((hsv[:,:,0] >= 18) & (hsv[:,:,0] <= 32) & 
                   (hsv[:,:,1] >= 80) & (hsv[:,:,2] >= 80))] = 1
    
    # Step 3: Prioritize white detection for driving lanes
    color_binary = np.zeros_like(gray)
    # Heavily weight white detection, minimize yellow impact
    color_binary[(white_mask == 1) | ((yellow_binary == 1) & (white_mask == 0))] = 1
    
    # Step 4: Gradient-based detection
    # Apply gaussian blur
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Sobel X gradient (emphasizes vertical lines)
    sobelx = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=9)
    abs_sobelx = np.absolute(sobelx)
    
    # Scale and threshold
    if np.max(abs_sobelx) > 0:
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        grad_binary = np.zeros_like(scaled_sobel)
        grad_binary[(scaled_sobel >= 35) & (scaled_sobel <= 255)] = 1  # Higher threshold for cleaner edges
    else:
        grad_binary = np.zeros_like(gray)
    
    # Step 5: Combine color and gradient
    combined_binary = np.zeros_like(gray)
    combined_binary[(color_binary == 1) | (grad_binary == 1)] = 1
    
    # Step 6: Clean up with morphological operations
    # Use smaller kernels to preserve lane line structure
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(combined_binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    # Convert back to 0-255 range
    result = cleaned * 255
    
    return result.astype(np.uint8)


def _x_at_y(line: Line, y: int) -> float:
    x1, y1, x2, y2 = line
    if y1 == y2:
        return float(x1)
    # Linear interpolation in (y -> x) space since lines are short segments
    t = (float(y) - y1) / (y2 - y1)
    return float(x1) + t * (x2 - x1)


def _order_left_right(lines: List[Line], y: int) -> Tuple[Optional[Line], Optional[Line]]:
    if not lines:
        return None, None
    if len(lines) == 1:
        return (lines[0], None) if _x_at_y(lines[0], y) < float('inf') else (None, lines[0])
    # pick two with smallest/ largest x at y
    xs = [(_x_at_y(l, y), l) for l in lines]
    xs.sort(key=lambda p: p[0])
    left = xs[0][1]
    right = xs[-1][1]
    return left, right


def estimate_lanes(
    seg: np.ndarray,
    depth: Optional[np.ndarray],
    bgr_image: Optional[np.ndarray] = None,
    target_width_in: Optional[float] = None,
    pixels_per_inch: Optional[float] = None,
    width_tolerance_in: float = 0.5,
) -> List[Line]:
    """
    Estimate lane lines and optionally enforce a target lane width (in inches) when a
    pixel-to-inch scale is provided.
    Returns a list of 2 line segments [(x1,y1,x2,y2), ...].
    """
    # Sliding-window polynomial fit for lane lines
    if bgr_image is None:
        # synthesize grayscale BGR
        bgr_image = cv2.cvtColor((seg.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
    h, w = bgr_image.shape[:2]

    binary = _binary_lane_mask(bgr_image)
    # Apply improved trapezoidal ROI that better matches road perspective
    roi_mask = np.zeros_like(binary)
    # More realistic trapezoid for highway perspective
    roi = np.array([[
        (int(0.1 * w), h),                    # bottom left
        (int(0.4 * w), int(0.65 * h)),        # top left
        (int(0.6 * w), int(0.65 * h)),        # top right
        (int(0.9 * w), h)                     # bottom right
    ]], dtype=np.int32)
    cv2.fillPoly(roi_mask, roi, 255)
    binary = cv2.bitwise_and(binary, roi_mask)

    # Simplified and accurate histogram analysis
    # Focus on bottom 30% of image where lane markings are most clear
    bottom_quarter = binary[int(h * 0.7):, :]
    histogram = np.sum(bottom_quarter, axis=0)
    
    # Find lane base positions using peaks in histogram
    midpoint = w // 2
    
    # Search for LEFT DRIVING LANE boundary (avoid center)
    left_search_end = int(0.48 * w)  # Stop before center to avoid yellow line
    left_half = histogram[int(0.1*w):left_search_end]
    if np.max(left_half) > 50:  # Higher threshold for confidence
        leftx_base = np.argmax(left_half) + int(0.1*w)
        # Ensure it's truly on the left side
        leftx_base = max(leftx_base, int(0.15 * w))
        leftx_base = min(leftx_base, int(0.42 * w))
    else:
        leftx_base = int(0.25 * w)  # Default left position
    
    # Search for RIGHT DRIVING LANE boundary (avoid center)
    right_search_start = int(0.52 * w)  # Start after center to avoid yellow line
    right_half = histogram[right_search_start:int(0.9*w)]
    if np.max(right_half) > 50:  # Higher threshold for confidence
        rightx_base = np.argmax(right_half) + right_search_start
        # Ensure it's truly on the right side
        rightx_base = max(rightx_base, int(0.58 * w))
        rightx_base = min(rightx_base, int(0.85 * w))
    else:
        rightx_base = int(0.75 * w)  # Default right position
    
    # Ensure reasonable driving lane width (not too narrow, not too wide)
    lane_width = rightx_base - leftx_base
    if lane_width < w * 0.15:  # Too narrow
        center = (leftx_base + rightx_base) // 2
        leftx_base = center - int(w * 0.1)
        rightx_base = center + int(w * 0.1)
    elif lane_width > w * 0.6:  # Too wide, probably picking up wrong markings
        center = (leftx_base + rightx_base) // 2
        leftx_base = center - int(w * 0.15)
        rightx_base = center + int(w * 0.15)

    # Sliding window search
    nwindows = 9
    window_height = int(h / nwindows)
    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Use smaller margin for more precise lane tracking
    margin = max(int(w * 0.04), 50)  # 4% of image width for precision
    minpix = 30  # Lower minimum pixels but with validation
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = h - (window + 1) * window_height
        win_y_high = h - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                         (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # Recenter with validation to prevent jumping to wrong markings
        if len(good_left_inds) > minpix:
            new_leftx = int(np.mean(nonzerox[good_left_inds]))
            # Only update if the change is reasonable (not jumping to center line)
            if abs(new_leftx - leftx_current) < w * 0.08:  # Max 8% jump
                leftx_current = new_leftx
        if len(good_right_inds) > minpix:
            new_rightx = int(np.mean(nonzerox[good_right_inds]))
            if abs(new_rightx - rightx_current) < w * 0.08:
                rightx_current = new_rightx

    left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([], dtype=int)
    right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([], dtype=int)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    lanes: List[Line] = []

    def fit_and_make_line(xs: np.ndarray, ys: np.ndarray) -> Optional[Line]:
        """Fit polynomial and create line"""
        if len(xs) < 30:  # Lower threshold for better detection
            return None
        
        try:
            # Fit a second order polynomial
            fit = np.polyfit(ys, xs, 2)
            
            # Generate y values for line
            y_bottom = h
            y_top = int(h * 0.6)
            
            # Calculate corresponding x values
            x_bottom = int(fit[0] * y_bottom**2 + fit[1] * y_bottom + fit[2])
            x_top = int(fit[0] * y_top**2 + fit[1] * y_top + fit[2])
            
            # Basic validation - ensure points are within image bounds
            if (0 <= x_bottom < w and 0 <= x_top < w):
                return (x_bottom, y_bottom, x_top, y_top)
            else:
                return None
                
        except (np.RankWarning, TypeError, ValueError):
            return None

    # Try to fit polynomials to left and right lane pixels
    left_line = fit_and_make_line(leftx, lefty)
    right_line = fit_and_make_line(rightx, righty)
    
    # Create output lane list
    lanes = []
    
    # Add detected lanes to output
    if left_line is not None:
        lanes.append(left_line)
    if right_line is not None:
        lanes.append(right_line)
    
    # If no lanes detected, use fallback based on histogram peaks
    if len(lanes) == 0:
        # Use histogram-based fallback
        left_fallback = (leftx_base, h, leftx_base, int(h * 0.6))
        right_fallback = (rightx_base, h, rightx_base, int(h * 0.6))
        lanes = [left_fallback, right_fallback]
    
    # If only one lane detected, synthesize the other
    elif len(lanes) == 1:
        detected_lane = lanes[0]
        x_bottom, y_bottom, x_top, y_top = detected_lane
        
        # Determine if it's left or right based on position
        if x_bottom < w // 2:  # It's a left lane
            # Synthesize right lane
            lane_width = w // 3  # Typical lane width
            right_synthetic = (x_bottom + lane_width, y_bottom, x_top + lane_width, y_top)
            lanes.append(right_synthetic)
        else:  # It's a right lane
            # Synthesize left lane
            lane_width = w // 3
            left_synthetic = (x_bottom - lane_width, y_bottom, x_top - lane_width, y_top)
            lanes.insert(0, left_synthetic)
    
    # Ensure lanes are ordered left to right
    if len(lanes) >= 2:
        lanes.sort(key=lambda lane: lane[0])  # Sort by bottom x position

    # Enforce target width if scale is provided
    if pixels_per_inch and target_width_in:
        target_dx_px = int(round(target_width_in * float(pixels_per_inch)))
        y_bottom = h
        # If only one line present, synthesize the counterpart at target width
        if left_line is None and right_line is not None:
            # Create left by subtracting target dx
            rx1, ry1, rx2, ry2 = right_line
            new_left = (rx1 - target_dx_px, ry1, rx2 - target_dx_px, ry2)
            lanes = [new_left, right_line]
            left_line = new_left
        elif right_line is None and left_line is not None:
            lx1, ly1, lx2, ly2 = left_line
            new_right = (lx1 + target_dx_px, ly1, lx2 + target_dx_px, ly2)
            lanes = [left_line, new_right]
            right_line = new_right
        elif left_line is not None and right_line is not None:
            # Adjust right line if measured width deviates beyond tolerance
            left_xb = _x_at_y(left_line, y_bottom)
            right_xb = _x_at_y(right_line, y_bottom)
            measured_dx_px = int(round(right_xb - left_xb))
            if measured_dx_px > 0 and pixels_per_inch > 0:
                measured_in = measured_dx_px / float(pixels_per_inch)
                if abs(measured_in - target_width_in) > float(width_tolerance_in):
                    # Shift right line to match the target width (keep its shape by offsetting x)
                    rx1, ry1, rx2, ry2 = right_line
                    shift = int(round(target_dx_px - measured_dx_px))
                    adjusted_right = (rx1 + shift, ry1, rx2 + shift, ry2)
                    lanes = [left_line, adjusted_right]
                    right_line = adjusted_right

    # Ensure output ordered left, right at bottom
    left, right = _order_left_right(lanes, h)
    lanes_out: List[Line] = []
    if left is not None:
        lanes_out.append(left)
    if right is not None and (not lanes_out or right != lanes_out[0]):
        lanes_out.append(right)
    return lanes_out
