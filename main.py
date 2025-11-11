#!/usr/bin/env python3
"""
Real-time Object & Person Detection with Lane Assignment
Integrates YOLOv8 detection with lane detection system
"""
import cv2
import numpy as np
import torch
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

# Import YOLOv8
from ultralytics import YOLO

# Import lane detection
from improved_accurate_lane_detection import ImprovedLaneDetector as LaneDetector


class ObjectTracker:
    """Simple object tracker using center point distance"""
    
    def __init__(self, max_disappeared: int = 30):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
    
    def register(self, centroid):
        """Register new object"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """Remove tracked object"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, detections):
        """Update tracked objects with new detections"""
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        input_centroids = np.array([d['center'] for d in detections])
        
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Compute distance between each pair
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            
            # Find minimum distance matches
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                if D[row, col] < 50:  # Maximum distance threshold
                    object_id = object_ids[row]
                    self.objects[object_id] = input_centroids[col]
                    self.disappeared[object_id] = 0
                    
                    used_rows.add(row)
                    used_cols.add(col)
            
            # Register new objects
            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col])
            
            # Mark disappeared objects
            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
        
        return self.objects


class DrivingAssistant:
    """Main class for object detection + lane integration"""
    
    # Target classes for detection
    DETECTION_CLASSES = {
        2: 'car',
        5: 'bus',
        3: 'motorcycle',
        7: 'truck',
        0: 'person'
    }
    
    # Colors for each class (BGR)
    CLASS_COLORS = {
        'car': (0, 255, 0),      # Green
        'bus': (255, 0, 0),      # Blue
        'motorcycle': (0, 165, 255),  # Orange
        'truck': (0, 0, 255),    # Red
        'person': (255, 0, 255)  # Magenta
    }
    
    def __init__(self, model_path: str = 'yolov8s.pt', device: str = 'auto', conf_threshold: float = 0.5):
        """
        Initialize detection system
        
        Args:
            model_path: Path to YOLO model weights
            device: 'cuda', 'cpu', or 'auto'
            conf_threshold: Confidence threshold for detections
        """
        self.conf_threshold = conf_threshold
        
        # Setup device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"🔧 Initializing Driving Assistant...")
        print(f"   Device: {self.device}")
        print(f"   Confidence Threshold: {conf_threshold}")
        
        # Load YOLO model
        self.model = self.load_model(model_path)
        
        # Initialize tracker
        self.tracker = ObjectTracker(max_disappeared=30)
        
        # Lane detector
        self.lane_detector = None
        
        # Statistics
        self.frame_count = 0
        self.fps = 0.0
        self.detection_stats = defaultdict(int)
    
    def load_model(self, model_path: str) -> YOLO:
        """Load YOLOv8 model"""
        print(f"📦 Loading YOLOv8 model: {model_path}")
        
        try:
            model = YOLO(model_path)
            
            # Set device
            if self.device == 'cuda':
                model.to('cuda')
                print("   ✓ Model loaded on GPU")
            else:
                print("   ✓ Model loaded on CPU")
            
            return model
            
        except Exception as e:
            print(f"   ✗ Error loading model: {e}")
            print("   → Downloading YOLOv8s model...")
            model = YOLO('yolov8s.pt')
            return model
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in frame
        
        Returns:
            List of detections with bbox, class, confidence, center
        """
        # Run inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        
        detections = []
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            
            # Filter only target classes
            if cls_id not in self.DETECTION_CLASSES:
                continue
            
            # Get bbox coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            
            # Calculate center
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            detection = {
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'class_id': cls_id,
                'class_name': self.DETECTION_CLASSES[cls_id],
                'confidence': conf,
                'center': (cx, cy)
            }
            
            detections.append(detection)
            self.detection_stats[self.DETECTION_CLASSES[cls_id]] += 1
        
        return detections
    
    def track_objects(self, detections: List[Dict]) -> List[Dict]:
        """
        Track objects across frames
        
        Returns:
            Detections with tracking IDs
        """
        tracked_objects = self.tracker.update(detections)
        
        # Match tracking IDs to detections
        for detection in detections:
            center = detection['center']
            # Find closest tracked object
            min_dist = float('inf')
            best_id = -1
            
            for obj_id, obj_center in tracked_objects.items():
                dist = np.linalg.norm(np.array(center) - np.array(obj_center))
                if dist < min_dist and dist < 50:
                    min_dist = dist
                    best_id = obj_id
            
            detection['track_id'] = best_id if best_id != -1 else -1
        
        return detections
    
    def point_in_polygon(self, point: Tuple[int, int], polygon: np.ndarray) -> bool:
        """Check if point is inside polygon"""
        return cv2.pointPolygonTest(polygon, point, False) >= 0
    
    def assign_lane_to_object(self, detections: List[Dict], lane_polygons: List[np.ndarray]) -> List[Dict]:
        """
        Assign each object to a lane
        
        Args:
            detections: List of detected objects
            lane_polygons: List of lane boundary polygons
            
        Returns:
            Detections with lane assignments
        """
        if not lane_polygons:
            for detection in detections:
                detection['lane'] = None
            return detections
        
        for detection in detections:
            center = detection['center']
            detection['lane'] = None
            
            # Check which lane the object center is in
            for i, polygon in enumerate(lane_polygons):
                if self.point_in_polygon(center, polygon):
                    detection['lane'] = i
                    break
        
        return detections
    
    def draw_results(self, frame: np.ndarray, detections: List[Dict], 
                    lane_overlay: np.ndarray = None) -> np.ndarray:
        """
        Draw detection results on frame
        
        Args:
            frame: Original frame
            detections: Detected objects with tracking and lane info
            lane_overlay: Lane detection overlay
            
        Returns:
            Annotated frame
        """
        output = frame.copy()
        
        # Apply lane overlay if available
        if lane_overlay is not None:
            output = cv2.addWeighted(output, 0.7, lane_overlay, 0.3, 0)
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            track_id = det.get('track_id', -1)
            lane = det.get('lane', None)
            cx, cy = det['center']
            
            # Get color for class
            color = self.CLASS_COLORS.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            cv2.circle(output, (cx, cy), 5, color, -1)
            
            # Prepare label
            label = f"{class_name} {confidence:.2f}"
            if track_id != -1:
                label += f" ID:{track_id}"
            if lane is not None:
                label += f" L{lane}"
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(output, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            
            # Draw label text
            cv2.putText(output, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return output
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process single frame - main pipeline
        
        Returns:
            (annotated_frame, detections)
        """
        # Detect lanes (if detector initialized)
        lane_overlay = None
        lane_polygons = []
        
        if self.lane_detector is not None:
            try:
                lane_result = self.lane_detector.process_frame(frame.copy())
                lane_overlay = lane_result
                # Extract lane polygons for assignment (simplified)
                # In production, extract actual lane boundaries from lane_detector
            except:
                pass
        
        # Detect objects
        detections = self.detect_objects(frame)
        
        # Track objects
        detections = self.track_objects(detections)
        
        # Assign lanes
        detections = self.assign_lane_to_object(detections, lane_polygons)
        
        # Draw results
        output = self.draw_results(frame, detections, lane_overlay)
        
        return output, detections
    
    def run(self, video_source, output_path: str = 'output.mp4', display: bool = True):
        """
        Main execution loop
        
        Args:
            video_source: Video file path or camera index (0 for webcam)
            output_path: Path to save output video
            display: Show live window
        """
        # Open video
        if isinstance(video_source, int):
            cap = cv2.VideoCapture(video_source)
            print(f"📹 Opening webcam {video_source}")
        else:
            cap = cv2.VideoCapture(video_source)
            print(f"📹 Opening video: {video_source}")
        
        if not cap.isOpened():
            print("❌ Error: Cannot open video source")
            return
        
        # Get video properties
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   Resolution: {width}x{height}")
        print(f"   Input FPS: {fps_in}")
        if total_frames > 0:
            print(f"   Total Frames: {total_frames}")
        
        # Initialize lane detector
        ret, first_frame = cap.read()
        if ret:
            self.lane_detector = LaneDetector(first_frame.shape)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            print("✓ Lane detector initialized")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))
        
        print(f"\n🚀 Starting detection...")
        print(f"   Output: {output_path}")
        print(f"   Display: {'ON' if display else 'OFF'}")
        print(f"   Press 'q' to quit\n")
        
        # Performance tracking
        frame_times = []
        start_time = time.time()
        
        try:
            while True:
                frame_start = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                output, detections = self.process_frame(frame)
                
                # Calculate FPS
                frame_time = time.time() - frame_start
                frame_times.append(frame_time)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                self.fps = 1.0 / np.mean(frame_times) if frame_times else 0
                
                # Draw FPS and stats
                cv2.putText(output, f"FPS: {self.fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(output, f"Frame: {self.frame_count}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(output, f"Objects: {len(detections)}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Write output
                out.write(output)
                self.frame_count += 1
                
                # Display
                if display:
                    cv2.imshow('Driving Assistant - Object & Lane Detection', output)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Progress update
                if self.frame_count % 100 == 0:
                    if total_frames > 0:
                        progress = (self.frame_count / total_frames) * 100
                        print(f"Progress: {progress:.1f}% | Frame {self.frame_count}/{total_frames} | FPS: {self.fps:.1f}")
                    else:
                        print(f"Frame {self.frame_count} | FPS: {self.fps:.1f}")
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            out.release()
            if display:
                cv2.destroyAllWindows()
            
            # Final statistics
            total_time = time.time() - start_time
            print(f"\n✅ Processing complete!")
            print(f"   Total frames: {self.frame_count}")
            print(f"   Total time: {total_time:.1f}s")
            print(f"   Average FPS: {self.frame_count/total_time:.1f}")
            print(f"   Output saved: {output_path}")
            
            print(f"\n📊 Detection Statistics:")
            for class_name, count in self.detection_stats.items():
                print(f"   {class_name}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Real-time Object & Person Detection with Lane Integration')
    parser.add_argument('--video', type=str, default=None, help='Path to video file')
    parser.add_argument('--cam', type=int, default=None, help='Camera index (e.g., 0 for webcam)')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output video path')
    parser.add_argument('--model', type=str, default='yolov8s.pt', help='YOLO model path')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], 
                       help='Device to use')
    parser.add_argument('--no-display', action='store_true', help='Disable display window')
    
    args = parser.parse_args()
    
    # Determine video source
    if args.cam is not None:
        video_source = args.cam
    elif args.video is not None:
        video_source = args.video
    else:
        # Default to project video
        video_source = 'project_video.mp4'
        print(f"ℹ️  No video source specified, using default: {video_source}")
    
    # Initialize system
    assistant = DrivingAssistant(
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf
    )
    
    # Run detection
    assistant.run(
        video_source=video_source,
        output_path=args.output,
        display=not args.no_display
    )


if __name__ == '__main__':
    main()
