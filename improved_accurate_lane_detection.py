#!/usr/bin/env python3
"""
Improved Accurate Lane Detection matching the desired output
"""
import cv2
import numpy as np


class ImprovedLaneDetector:
    def __init__(self, img_shape):
        self.h, self.w = img_shape[:2]
        self.prev_left_fit = None
        self.prev_right_fit = None
        self.lane_center_offset = 0
        
        # Real-world conversion factors (US highway standard)
        # Assuming lane width is ~3.7 meters (12 feet)
        self.ym_per_pix = 30 / 720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        
        # Accuracy tracking
        self.total_frames = 0
        self.successful_detections = 0
        self.lane_detection_accuracy = 0.0
        
        self.setup_perspective_transform()
    
    def setup_perspective_transform(self):
        """Setup perspective transform for bird's eye view"""
        # Source points - adjusted for better accuracy
        src = np.float32([
            [self.w * 0.465, self.h * 0.62],  # top left
            [self.w * 0.535, self.h * 0.62],  # top right
            [self.w * 0.90, self.h * 0.96],   # bottom right
            [self.w * 0.10, self.h * 0.96]    # bottom left
        ])
        
        # Destination points
        dst = np.float32([
            [self.w * 0.20, 0],
            [self.w * 0.80, 0],
            [self.w * 0.80, self.h],
            [self.w * 0.20, self.h]
        ])
        
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
    
    def detect_lane_pixels(self, img):
        """Enhanced lane pixel detection with better white/yellow detection"""
        # Convert to different color spaces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # White lane detection - improved thresholds
        white_binary = cv2.inRange(gray, 180, 255)
        white_hls_l = hls[:, :, 1]
        white_hls = cv2.inRange(white_hls_l, 180, 255)
        white_combined = cv2.bitwise_or(white_binary, white_hls)
        
        # Yellow lane detection - better for center lines
        yellow_hsv = cv2.inRange(hsv, (18, 100, 100), (30, 255, 255))
        yellow_hls_h = hls[:, :, 0]
        yellow_hls_s = hls[:, :, 2]
        yellow_hls = cv2.inRange(hls, (18, 80, 80), (30, 255, 255))
        yellow_combined = cv2.bitwise_or(yellow_hsv, yellow_hls)
        
        # Sobel edge detection for lane edges
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=7)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        sobel_binary = cv2.inRange(scaled_sobel, 25, 255)
        
        # Combine all detections
        combined = np.zeros_like(gray)
        combined[(white_combined > 0) | (yellow_combined > 0) | (sobel_binary > 0)] = 255
        
        # Apply region of interest mask
        mask = np.zeros_like(combined)
        roi_vertices = np.array([[(0, self.h), (self.w * 0.45, self.h * 0.60), 
                                  (self.w * 0.55, self.h * 0.60), (self.w, self.h)]], dtype=np.int32)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked = cv2.bitwise_and(combined, mask)
        
        return masked
    
    def find_lane_pixels_sliding_window(self, binary_warped):
        """Improved sliding window with better peak detection"""
        # Histogram of bottom quarter (more focused)
        histogram = np.sum(binary_warped[int(binary_warped.shape[0] * 0.75):, :], axis=0)
        
        # Smooth histogram
        histogram = np.convolve(histogram, np.ones(20) / 20, mode='same')
        
        # Find peaks
        midpoint = len(histogram) // 2
        
        # Search for left lane in left half
        left_half = histogram[:midpoint]
        if np.max(left_half) > 100:
            leftx_base = np.argmax(left_half)
        else:
            leftx_base = midpoint // 2
        
        # Search for right lane in right half
        right_half = histogram[midpoint:]
        if np.max(right_half) > 100:
            rightx_base = np.argmax(right_half) + midpoint
        else:
            rightx_base = midpoint + midpoint // 2
        
        # Window settings - optimized for accuracy
        nwindows = 12
        margin = 80
        minpix = 40
        
        window_height = binary_warped.shape[0] // nwindows
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        left_lane_inds = []
        right_lane_inds = []
        
        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            
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
            
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
        
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        return leftx, lefty, rightx, righty
    
    def fit_polynomial(self, leftx, lefty, rightx, righty):
        """Fit polynomial with temporal smoothing"""
        detection_successful = False
        
        # Fit left lane
        if len(leftx) > 150:
            left_fit = np.polyfit(lefty, leftx, 2)
            # Temporal smoothing
            if self.prev_left_fit is not None:
                left_fit = 0.75 * self.prev_left_fit + 0.25 * left_fit
            self.prev_left_fit = left_fit
            detection_successful = True
        else:
            left_fit = self.prev_left_fit if self.prev_left_fit is not None else [0, 0, self.w // 4]
        
        # Fit right lane
        if len(rightx) > 150:
            right_fit = np.polyfit(righty, rightx, 2)
            # Temporal smoothing
            if self.prev_right_fit is not None:
                right_fit = 0.75 * self.prev_right_fit + 0.25 * right_fit
            self.prev_right_fit = right_fit
            if detection_successful:
                detection_successful = True
        else:
            right_fit = self.prev_right_fit if self.prev_right_fit is not None else [0, 0, 3 * self.w // 4]
            detection_successful = False
        
        # Update accuracy tracking
        self.total_frames += 1
        if detection_successful:
            self.successful_detections += 1
        
        # Calculate accuracy
        if self.total_frames > 0:
            self.lane_detection_accuracy = (self.successful_detections / self.total_frames) * 100
        
        return left_fit, right_fit
    
    def calculate_lane_measurements(self, left_fit, right_fit, img_height):
        """Calculate lane measurements including width and distance to edges"""
        # Calculate lane positions at bottom of image
        y_eval = img_height - 1
        left_lane_bottom = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
        right_lane_bottom = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]
        
        # Lane center
        lane_center = (left_lane_bottom + right_lane_bottom) / 2
        
        # Vehicle center (assume camera is at center)
        vehicle_center = self.w / 2
        
        # Offset in pixels (positive = vehicle is right of center, negative = left of center)
        offset_pixels = vehicle_center - lane_center
        
        # Calculate measurements in meters
        lane_width_pixels = right_lane_bottom - left_lane_bottom
        lane_width_meters = lane_width_pixels * self.xm_per_pix
        
        # Distance to left and right edges
        distance_to_left = (vehicle_center - left_lane_bottom) * self.xm_per_pix
        distance_to_right = (right_lane_bottom - vehicle_center) * self.xm_per_pix
        
        # Offset from center in meters
        offset_meters = offset_pixels * self.xm_per_pix
        
        # Calculate curvature radius
        curvature_radius = self.calculate_curvature(left_fit, right_fit, img_height)
        
        return {
            'offset_pixels': offset_pixels,
            'offset_meters': offset_meters,
            'lane_width': lane_width_meters,
            'distance_to_left': distance_to_left,
            'distance_to_right': distance_to_right,
            'curvature_radius': curvature_radius
        }
    
    def calculate_curvature(self, left_fit, right_fit, img_height):
        """Calculate radius of curvature of the lane"""
        y_eval = img_height - 1
        
        # Convert polynomial coefficients to real-world space
        left_fit_cr = np.polyfit(np.array([0, img_height]) * self.ym_per_pix, 
                                 np.array([left_fit[2], left_fit[0] * img_height**2 + 
                                          left_fit[1] * img_height + left_fit[2]]) * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(np.array([0, img_height]) * self.ym_per_pix,
                                  np.array([right_fit[2], right_fit[0] * img_height**2 + 
                                           right_fit[1] * img_height + right_fit[2]]) * self.xm_per_pix, 2)
        
        # Calculate radius of curvature
        y_eval_m = y_eval * self.ym_per_pix
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval_m + left_fit_cr[1])**2)**1.5) / np.abs(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval_m + right_fit_cr[1])**2)**1.5) / np.abs(2 * right_fit_cr[0])
        
        # Average curvature
        curvature = (left_curverad + right_curverad) / 2
        
        return curvature
    
    def calculate_recommended_speed(self, curvature_radius, lane_width, offset_meters):
        """Calculate recommended speed based on road conditions"""
        # Base speed on curvature
        if curvature_radius > 5000:  # Nearly straight
            base_speed = 120  # km/h
            road_type = "Straight"
        elif curvature_radius > 1500:  # Gentle curve
            base_speed = 100
            road_type = "Gentle Curve"
        elif curvature_radius > 800:  # Moderate curve
            base_speed = 80
            road_type = "Moderate Curve"
        elif curvature_radius > 400:  # Sharp curve
            base_speed = 60
            road_type = "Sharp Curve"
        else:  # Very sharp curve
            base_speed = 40
            road_type = "Sharp Curve"
        
        # Adjust for lane position (if drifting, reduce speed)
        if abs(offset_meters) > 0.5:
            base_speed *= 0.85  # Reduce by 15%
        
        # Adjust for narrow lanes
        if lane_width < 3.2:
            base_speed *= 0.9  # Reduce by 10%
        
        return int(base_speed), road_type
    
    def draw_speed_recommendation_panel(self, img, measurements):
        """Draw speed recommendation panel"""
        # Panel position (center top) - ultra compact size
        panel_w = 140
        panel_h = 55
        panel_x = (self.w - panel_w) // 2  # Center horizontally
        panel_y = 20
        
        curvature = measurements['curvature_radius']
        recommended_speed, road_type = self.calculate_recommended_speed(
            curvature, measurements['lane_width'], measurements['offset_meters']
        )
        
        # Create darker, more opaque panel background
        overlay = img.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h),
                     (10, 10, 20), -1)  # Much darker background
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h),
                     (180, 180, 180), 2)  # Gray border
        img = cv2.addWeighted(img, 0.3, overlay, 0.7, 0)  # More opaque
        
        # Title (very small)
        cv2.putText(img, "SPEED", (panel_x + 5, panel_y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Recommended speed (compact inline display)
        speed_color = (0, 255, 0) if recommended_speed >= 80 else (0, 255, 255) if recommended_speed >= 60 else (0, 165, 255)
        cv2.putText(img, f"{recommended_speed}", 
                   (panel_x + 10, panel_y + 42),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, speed_color, 2)
        cv2.putText(img, "km/h", 
                   (panel_x + 60, panel_y + 42),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return img
    
    def draw_lane_measurements_panel(self, img, measurements):
        """Draw lane measurements panel with distance and width info"""
        # Panel position (top-right area)
        panel_x = self.w - 380
        panel_y = 20
        panel_w = 360
        panel_h = 180
        
        # Create semi-transparent panel background with gradient effect
        overlay = img.copy()
        
        # Inner dark background
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h),
                     (20, 20, 30), -1)
        
        # Outer glow border (multiple rectangles for glow effect)
        cv2.rectangle(overlay, (panel_x - 2, panel_y - 2), 
                     (panel_x + panel_w + 2, panel_y + panel_h + 2),
                     (0, 255, 255), 3)
        cv2.rectangle(overlay, (panel_x - 1, panel_y - 1), 
                     (panel_x + panel_w + 1, panel_y + panel_h + 1),
                     (0, 200, 200), 2)
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h),
                     (0, 255, 255), 2)
        
        img = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
        
        # Title with glow effect
        cv2.putText(img, "LANE MEASUREMENTS", (panel_x + 11, panel_y + 31),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 150), 3)  # Shadow
        cv2.putText(img, "LANE MEASUREMENTS", (panel_x + 10, panel_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # Main text
        
        # Draw separator line
        cv2.line(img, (panel_x + 10, panel_y + 40), 
                (panel_x + panel_w - 10, panel_y + 40), (100, 100, 100), 1)
        
        # Lane width
        lane_width = measurements['lane_width']
        cv2.putText(img, f"Lane Width: {lane_width:.2f} m", 
                   (panel_x + 15, panel_y + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Distance to left edge
        dist_left = measurements['distance_to_left']
        color_left = (0, 255, 255) if dist_left > 0.5 else (0, 165, 255)  # Yellow if safe, orange if close
        cv2.putText(img, f"Left Edge: {dist_left:.2f} m", 
                   (panel_x + 15, panel_y + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_left, 2)
        
        # Distance to right edge
        dist_right = measurements['distance_to_right']
        color_right = (0, 255, 0) if dist_right > 0.5 else (0, 165, 255)  # Green if safe, orange if close
        cv2.putText(img, f"Right Edge: {dist_right:.2f} m", 
                   (panel_x + 15, panel_y + 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_right, 2)
        
        # Center offset
        offset = measurements['offset_meters']
        direction = "RIGHT" if offset > 0 else "LEFT" if offset < 0 else "CENTER"
        offset_color = (0, 255, 0) if abs(offset) < 0.2 else (0, 255, 255)
        cv2.putText(img, f"Offset: {abs(offset):.2f} m {direction}", 
                   (panel_x + 15, panel_y + 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, offset_color, 2)
        
        return img
    
    def draw_direction_arrow(self, img, offset_pixels):
        """Draw directional arrow based on lane offset"""
        # Threshold for showing arrow (in pixels)
        threshold = 30
        
        # Arrow position
        arrow_y = int(self.h * 0.3)  # 30% from top
        arrow_x = int(self.w / 2)
        
        if abs(offset_pixels) > threshold:
            if offset_pixels > 0:  # Vehicle is right of center, show left arrow
                direction = "LEFT"
                # Draw left arrow
                arrow_color = (0, 255, 255)  # Yellow
                cv2.arrowedLine(img, (arrow_x + 80, arrow_y), (arrow_x + 20, arrow_y),
                              arrow_color, 8, tipLength=0.5)
                cv2.putText(img, "STEER LEFT", (arrow_x - 60, arrow_y + 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, arrow_color, 3)
            else:  # Vehicle is left of center, show right arrow
                direction = "RIGHT"
                # Draw right arrow
                arrow_color = (0, 255, 255)  # Yellow
                cv2.arrowedLine(img, (arrow_x - 80, arrow_y), (arrow_x - 20, arrow_y),
                              arrow_color, 8, tipLength=0.5)
                cv2.putText(img, "STEER RIGHT", (arrow_x - 70, arrow_y + 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, arrow_color, 3)
        else:
            # Centered - show checkmark or "OK"
            cv2.putText(img, "CENTERED", (arrow_x - 90, arrow_y),
                      cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        return img
    
    def create_birds_eye_view(self, binary_warped, left_fit, right_fit, size=(200, 280)):
        """Create mini bird's eye view visualization"""
        mini_w, mini_h = size
        
        # Create blank canvas
        birds_eye = np.zeros((mini_h, mini_w, 3), dtype=np.uint8)
        birds_eye[:] = (40, 40, 40)  # Dark gray background
        
        # Generate y values for the mini view
        ploty = np.linspace(0, mini_h - 1, mini_h)
        
        # Scale factors to fit warped view into mini view
        scale_x = mini_w / binary_warped.shape[1]
        scale_y = mini_h / binary_warped.shape[0]
        
        # Calculate lane positions scaled to mini view
        left_fitx = (left_fit[0] * (ploty / scale_y)**2 + 
                     left_fit[1] * (ploty / scale_y) + 
                     left_fit[2]) * scale_x
        right_fitx = (right_fit[0] * (ploty / scale_y)**2 + 
                      right_fit[1] * (ploty / scale_y) + 
                      right_fit[2]) * scale_x
        
        # Clip values to prevent out of bounds
        left_fitx = np.clip(left_fitx, 0, mini_w - 1)
        right_fitx = np.clip(right_fitx, 0, mini_w - 1)
        
        # Create lane polygon
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))], dtype=np.int32)
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))], dtype=np.int32)
        pts = np.hstack((pts_left, pts_right))
        
        # Fill lane area (green)
        cv2.fillPoly(birds_eye, pts, (0, 200, 0))
        
        # Draw lane edges
        for i in range(len(ploty) - 1):
            # Left edge (yellow)
            cv2.line(birds_eye,
                    (int(left_fitx[i]), int(ploty[i])),
                    (int(left_fitx[i + 1]), int(ploty[i + 1])),
                    (0, 255, 255), 2)
            # Right edge (red)
            cv2.line(birds_eye,
                    (int(right_fitx[i]), int(ploty[i])),
                    (int(right_fitx[i + 1]), int(ploty[i + 1])),
                    (0, 0, 255), 2)
        
        # Draw vehicle position indicator (white triangle at bottom center)
        vehicle_x = mini_w // 2
        vehicle_y = mini_h - 20
        triangle = np.array([
            [vehicle_x, vehicle_y - 15],
            [vehicle_x - 10, vehicle_y + 5],
            [vehicle_x + 10, vehicle_y + 5]
        ], dtype=np.int32)
        cv2.fillPoly(birds_eye, [triangle], (255, 255, 255))
        cv2.polylines(birds_eye, [triangle], True, (0, 255, 0), 2)
        
        # Draw center line (dashed)
        center_x = mini_w // 2
        for y in range(0, mini_h, 20):
            cv2.line(birds_eye, (center_x, y), (center_x, y + 10), (150, 150, 150), 1)
        
        # Add border
        cv2.rectangle(birds_eye, (0, 0), (mini_w - 1, mini_h - 1), (255, 255, 255), 2)
        
        # Add title
        cv2.putText(birds_eye, "Bird's Eye View", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return birds_eye
    
    def draw_lane(self, img, binary_warped, left_fit, right_fit):
        """Draw detected lane with gradient edges matching desired output"""
        # Generate y values
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        
        # Calculate x values
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        
        # Calculate lane measurements
        measurements = self.calculate_lane_measurements(left_fit, right_fit, binary_warped.shape[0])
        self.lane_center_offset = measurements['offset_pixels']
        
        # Create overlay
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Create points for lane area
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Fill lane area with green
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        
        # Draw left edge with gradient (Yellow to Orange) - thicker and brighter
        for i in range(len(ploty) - 1):
            # Yellow line on left
            cv2.line(color_warp,
                    (int(left_fitx[i]), int(ploty[i])),
                    (int(left_fitx[i + 1]), int(ploty[i + 1])),
                    (0, 255, 255), 20)
        
        # Draw right edge with gradient (White to Red) - thicker and brighter
        for i in range(len(ploty) - 1):
            # Red line on right
            cv2.line(color_warp,
                    (int(right_fitx[i]), int(ploty[i])),
                    (int(right_fitx[i + 1]), int(ploty[i + 1])),
                    (0, 0, 255), 20)
        
        # Warp back to original perspective
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (img.shape[1], img.shape[0]))
        
        # Blend with original image - increase lane overlay visibility
        result = cv2.addWeighted(img, 1, newwarp, 0.5, 0)
        
        # Create and overlay mini bird's eye view
        birds_eye = self.create_birds_eye_view(binary_warped, left_fit, right_fit)
        
        # Position in bottom left corner (to avoid measurements panel)
        bev_h, bev_w = birds_eye.shape[:2]
        margin = 20
        y_offset = self.h - bev_h - margin
        x_offset = margin
        
        # Add semi-transparent background
        overlay = result.copy()
        cv2.rectangle(overlay, 
                     (x_offset - 5, y_offset - 5), 
                     (x_offset + bev_w + 5, y_offset + bev_h + 5),
                     (0, 0, 0), -1)
        result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
        
        # Overlay the bird's eye view
        result[y_offset:y_offset + bev_h, x_offset:x_offset + bev_w] = birds_eye
        
        # Draw lane measurements panel
        result = self.draw_lane_measurements_panel(result, measurements)
        
        # Draw speed recommendation panel
        result = self.draw_speed_recommendation_panel(result, measurements)
        
        return result
    
    def process_frame(self, img):
        """Process single frame"""
        # Detect lane pixels
        binary = self.detect_lane_pixels(img)
        
        # Apply perspective transform
        binary_warped = cv2.warpPerspective(binary, self.M, (img.shape[1], img.shape[0]),
                                           flags=cv2.INTER_LINEAR)
        
        # Find lane pixels
        leftx, lefty, rightx, righty = self.find_lane_pixels_sliding_window(binary_warped)
        
        # Fit polynomial
        left_fit, right_fit = self.fit_polynomial(leftx, lefty, rightx, righty)
        
        # Draw lane
        result = self.draw_lane(img, binary_warped, left_fit, right_fit)
        
        return result
    
    def get_accuracy(self):
        """Get current lane detection accuracy"""
        return self.lane_detection_accuracy
