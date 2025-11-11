#!/usr/bin/env python3
"""
Accurate Lane Detection with Perspective Transform (Bird's Eye View)
Industry-standard approach for 90%+ accuracy
"""
import cv2
import numpy as np
import argparse
import time


class LaneDetector:
    def __init__(self, img_shape):
        self.h, self.w = img_shape[:2]
        self.prev_left_fit = None
        self.prev_right_fit = None
        self.setup_perspective_transform()
    
    def setup_perspective_transform(self):
        """Setup perspective transform for bird's eye view"""
        # Source points (trapezoid in original view)
        src = np.float32([
            [self.w * 0.45, self.h * 0.63],  # top left
            [self.w * 0.55, self.h * 0.63],  # top right
            [self.w * 0.9, self.h],          # bottom right
            [self.w * 0.1, self.h]           # bottom left
        ])
        
        # Destination points (rectangle in bird's eye view)
        dst = np.float32([
            [self.w * 0.2, 0],
            [self.w * 0.8, 0],
            [self.w * 0.8, self.h],
            [self.w * 0.2, self.h]
        ])
        
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
    
    def detect_lane_pixels(self, img):
        """Enhanced lane pixel detection"""
        # Convert to different color spaces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # White lane detection (multiple methods)
        white_binary = cv2.inRange(gray, 200, 255)
        white_hls = cv2.inRange(hls, (0, 200, 0), (255, 255, 255))
        white_combined = cv2.bitwise_or(white_binary, white_hls)
        
        # Yellow lane detection (for center lines)
        yellow_hsv = cv2.inRange(hsv, (15, 80, 100), (30, 255, 255))
        yellow_hls = cv2.inRange(hls, (15, 30, 100), (30, 255, 255))
        yellow_combined = cv2.bitwise_or(yellow_hsv, yellow_hls)
        
        # Sobel edge detection (X direction - vertical lines)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=9)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        sobel_binary = cv2.inRange(scaled_sobel, 30, 255)
        
        # Combine all detections
        combined = np.zeros_like(gray)
        combined[(white_combined > 0) | (yellow_combined > 0) | (sobel_binary > 0)] = 255
        
        return combined
    
    def find_lane_pixels_sliding_window(self, binary_warped):
        """Sliding window search from scratch"""
        # Histogram of bottom half
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        
        # Find peaks
        midpoint = len(histogram) // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Window settings
        nwindows = 9
        margin = 100
        minpix = 50
        
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
        """Fit 2nd order polynomial with smoothing"""
        if len(leftx) > 100:
            left_fit = np.polyfit(lefty, leftx, 2)
            # Smooth with previous frame
            if self.prev_left_fit is not None:
                left_fit = 0.7 * self.prev_left_fit + 0.3 * left_fit
            self.prev_left_fit = left_fit
        else:
            left_fit = self.prev_left_fit if self.prev_left_fit is not None else [0, 0, self.w//4]
        
        if len(rightx) > 100:
            right_fit = np.polyfit(righty, rightx, 2)
            # Smooth with previous frame
            if self.prev_right_fit is not None:
                right_fit = 0.7 * self.prev_right_fit + 0.3 * right_fit
            self.prev_right_fit = right_fit
        else:
            right_fit = self.prev_right_fit if self.prev_right_fit is not None else [0, 0, 3*self.w//4]
        
        return left_fit, right_fit
    
    def draw_lane(self, img, binary_warped, left_fit, right_fit):
        """Draw detected lane area"""
        # Generate y values
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        
        # Calculate x values from polynomial
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Recast x and y for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        
        # Draw lane lines
        for i in range(len(ploty)-1):
            cv2.line(color_warp, 
                    (int(left_fitx[i]), int(ploty[i])), 
                    (int(left_fitx[i+1]), int(ploty[i+1])), 
                    (255, 0, 0), 10)
            cv2.line(color_warp, 
                    (int(right_fitx[i]), int(ploty[i])), 
                    (int(right_fitx[i+1]), int(ploty[i+1])), 
                    (0, 0, 255), 10)
        
        # Warp back to original image space
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (img.shape[1], img.shape[0]))
        
        # Combine with original image
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
        
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='project_video.mp4')
    parser.add_argument('--output', default='accurate_lanes.mp4')
    parser.add_argument('--display', action='store_true')
    args = parser.parse_args()
    
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Cannot open {args.source}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read first frame to initialize detector
    ret, first_frame = cap.read()
    if not ret:
        print("Cannot read video")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Initialize detector
    detector = LaneDetector(first_frame.shape)
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
    
    frame_count = 0
    start_time = time.time()
    
    print(f"Processing {total_frames} frames from {args.source}...")
    print("Using Perspective Transform (Bird's Eye View) for accurate detection")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        result = detector.process_frame(frame)
        
        # Add text overlay
        cv2.putText(result, f"Frame: {frame_count}/{total_frames}", 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(result, "Accurate Lane Detection (Bird's Eye View)", 
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        out.write(result)
        frame_count += 1
        
        if args.display:
            cv2.imshow('Accurate Lane Detection', result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% | {frame_count}/{total_frames} frames | FPS: {fps_actual:.1f}")
    
    cap.release()
    out.release()
    if args.display:
        cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    print(f"\n✅ Done! Saved to {args.output}")
    print(f"Total frames: {frame_count}")
    print(f"Processing time: {total_time:.1f}s")
    print(f"Average FPS: {frame_count/total_time:.1f}")


if __name__ == '__main__':
    main()
