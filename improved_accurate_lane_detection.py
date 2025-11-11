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
        # Fit left lane
        if len(leftx) > 150:
            left_fit = np.polyfit(lefty, leftx, 2)
            # Temporal smoothing
            if self.prev_left_fit is not None:
                left_fit = 0.75 * self.prev_left_fit + 0.25 * left_fit
            self.prev_left_fit = left_fit
        else:
            left_fit = self.prev_left_fit if self.prev_left_fit is not None else [0, 0, self.w // 4]
        
        # Fit right lane
        if len(rightx) > 150:
            right_fit = np.polyfit(righty, rightx, 2)
            # Temporal smoothing
            if self.prev_right_fit is not None:
                right_fit = 0.75 * self.prev_right_fit + 0.25 * right_fit
            self.prev_right_fit = right_fit
        else:
            right_fit = self.prev_right_fit if self.prev_right_fit is not None else [0, 0, 3 * self.w // 4]
        
        return left_fit, right_fit
    
    def draw_lane(self, img, binary_warped, left_fit, right_fit):
        """Draw detected lane with gradient edges matching desired output"""
        # Generate y values
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        
        # Calculate x values
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        
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
