import argparse
from collections import defaultdict, deque
import json
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO

import cv2
import math
import numpy as np

import supervision as sv


# --- Helpers: centroid and orientation extraction/drawing ---
def centroid_from_xyxy(xyxy: np.ndarray):
    """حساب centroid من bounding box"""
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return int(cx), int(cy)


def orientation_from_mask(mask: np.ndarray):
    """حساب orientation من mask باستخدام minAreaRect"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 10:
        return None
    (cx, cy), (w, h), angle = cv2.minAreaRect(cnt)
    if w < h:
        angle = angle + 90.0
    angle = (angle + 360.0) % 360.0
    return float(angle)


def orientation_from_points(points: np.ndarray):
    """حساب orientation باستخدام PCA على النقاط الأخيرة"""
    if points is None or len(points) < 3:
        return None
    pts = np.asarray(points, dtype=np.float32)
    mean = pts.mean(axis=0)
    centered = pts - mean
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    except Exception:
        return None
    principal = vt[0]
    angle_rad = math.atan2(principal[1], principal[0])
    angle_deg = math.degrees(angle_rad)
    angle_deg = (angle_deg + 360.0) % 360.0
    return float(angle_deg)


def calculate_slip_angle(heading_deg: float, motion_deg: float):
    """
    حساب Slip Angle: الفرق بين اتجاه السيارة واتجاه الحركة
    نرجع أصغر زاوية (0-180)
    """
    if heading_deg is None or motion_deg is None:
        return None
    
    diff = abs(heading_deg - motion_deg)
    # نأخذ أصغر زاوية (مثلاً 350° = 10°)
    if diff > 180:
        diff = 360 - diff
    return diff


def classify_behavior(slip_angle: float):
    """
    تصنيف السلوك حسب Slip Angle
    """
    if slip_angle is None:
        return "Unknown"
    
    if slip_angle < 10:
        return "Normal"
    elif slip_angle < 20:
        return "Slight Zigzag"
    elif slip_angle < 35:
        return "Zigzag"
    elif slip_angle < 60:
        return "Drift"
    else:
        return "Extreme Drift"


def get_behavior_color(behavior: str):
    """لون حسب التصنيف"""
    colors = {
        "Normal": (0, 255, 0),        # أخضر
        "Slight Zigzag": (0, 255, 255),  # أصفر
        "Zigzag": (0, 165, 255),      # برتقالي
        "Drift": (0, 100, 255),       # برتقالي غامق
        "Extreme Drift": (0, 0, 255), # أحمر
        "Unknown": (128, 128, 128)    # رمادي
    }
    return colors.get(behavior, (255, 255, 255))


def draw_centroid_and_orientation(frame: np.ndarray, centroid, angle_deg, length: int = 40, color=(0, 255, 255)):
    """رسم centroid و orientation arrow"""
    cx, cy = int(centroid[0]), int(centroid[1])
    cv2.circle(frame, (cx, cy), radius=4, color=(0, 0, 255), thickness=-1)
    if angle_deg is None:
        return
    angle_rad = math.radians(angle_deg)
    x2 = int(cx + length * math.cos(angle_rad))
    y2 = int(cy + length * math.sin(angle_rad))
    cv2.arrowedLine(frame, (cx, cy), (x2, y2), color=color, thickness=2, tipLength=0.3)


def draw_dashboard(frame: np.ndarray, tracker_stats: dict, frame_number: int):
    """رسم dashboard يعرض الإحصائيات بحجم أصغر"""
    h, w = frame.shape[:2]
    
    # Background panel - smaller size and more transparent
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (220, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Title - smaller font
    cv2.putText(frame, "Drift Detection", (15, 26), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    cv2.line(frame, (15, 32), (210, 32), (255, 255, 255), 1)
    
    # Frame number - smaller font
    cv2.putText(frame, f"Frame: {frame_number}", (15, 47), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    
    # Active vehicles - smaller font
    cv2.putText(frame, f"Active Cars: {len(tracker_stats)}", (15, 61), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    
    # Behavior counts - smaller font
    y_offset = 76
    behavior_counts = defaultdict(int)
    for stats in tracker_stats.values():
        behavior_counts[stats['current_behavior']] += 1
    
    for behavior, count in sorted(behavior_counts.items()):
        if count > 0:
            color = get_behavior_color(behavior)
            cv2.putText(frame, f"{behavior}: {count}", (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)
            y_offset += 15


# -----------------------------------------------------------


def process_video(
    source_weights_path: str,
    source_video_path: str,
    target_video_path: str,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
    output_stats: bool = True,
) -> None:
    """
    معالجة الفيديو مع detection الانحراف والـ zigzag - للسيارات فقط
    """
    model = YOLO(source_weights_path)

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    # تخزين البيانات لكل tracker
    coordinates = defaultdict(lambda: deque(maxlen=int(30 * 5)))  # 5 seconds at 30fps
    slip_angles_history = defaultdict(lambda: deque(maxlen=30))  # آخر ثانية
    tracker_statistics = defaultdict(lambda: {
        'behavior_counts': defaultdict(int),
        'max_slip_angle': 0,
        'avg_slip_angle': 0,
        'total_frames': 0,
        'current_behavior': 'Unknown'
    })
    
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)

    with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
        prev_gray = None
        frame_number = 0
        
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            frame_number += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # YOLO Detection
            results = model(
                frame, verbose=False, conf=confidence_threshold, iou=iou_threshold
            )[0]
            detections = sv.Detections.from_ultralytics(results)
            
            # Filter only cars (class 2 in COCO dataset)
            # COCO classes: 2 = car, 5 = bus, 7 = truck
            # Keep only cars (class 2)
            car_mask = detections.class_id == 2
            detections = detections[car_mask]
            
            detections = tracker.update_with_detections(detections)
            
            # Get anchors
            try:
                anchors = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            except Exception:
                anchors = np.array([])

            if anchors is not None and anchors.size:
                for tracker_id, (x, y) in zip(detections.tracker_id, anchors):
                    coordinates[tracker_id].append((int(x), int(y)))

            # Annotate frame
            annotated_frame = box_annotator.annotate(
                scene=frame.copy(), detections=detections
            )
            annotated_labeled_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections
            )

            # Extract boxes
            boxes = None
            if hasattr(detections, "xyxy"):
                try:
                    boxes = np.array(detections.xyxy)
                except Exception:
                    boxes = None

            masks = getattr(detections, "masks", None)

            # Process each detection
            for idx, tracker_id in enumerate(detections.tracker_id):
                centroid = None
                if boxes is not None and idx < len(boxes):
                    centroid = centroid_from_xyxy(boxes[idx])

                # Get heading angle (orientation)
                heading_angle = None
                if masks is not None:
                    try:
                        mask_i = masks[idx]
                        if hasattr(mask_i, "shape") and mask_i.shape[:2] == frame.shape[:2]:
                            heading_angle = orientation_from_mask(mask_i)
                    except Exception:
                        heading_angle = None

                # Fallback: PCA
                if heading_angle is None:
                    pts = coordinates.get(tracker_id, None)
                    if pts is not None and len(pts) >= 3:
                        heading_angle = orientation_from_points(np.array(pts))

                # Calculate motion direction using optical flow
                motion_angle = None
                slip_angle = None
                
                if centroid is not None and prev_gray is not None and boxes is not None and idx < len(boxes):
                    x1, y1, x2, y2 = boxes[idx].astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
                    
                    if x2 - x1 > 4 and y2 - y1 > 4:
                        prev_roi = prev_gray[y1:y2, x1:x2]
                        curr_roi = gray[y1:y2, x1:x2]
                        
                        try:
                            flow = cv2.calcOpticalFlowFarneback(
                                prev_roi, curr_roi, None,
                                0.5, 3, 15, 3, 5, 1.2, 0
                            )
                            
                            mean_u = float(np.mean(flow[..., 0]))
                            mean_v = float(np.mean(flow[..., 1]))
                            
                            # حساب اتجاه الحركة
                            motion_angle = math.degrees(math.atan2(mean_v, mean_u))
                            motion_angle = (motion_angle + 360.0) % 360.0
                            
                            # حساب Slip Angle
                            if heading_angle is not None:
                                slip_angle = calculate_slip_angle(heading_angle, motion_angle)
                                slip_angles_history[tracker_id].append(slip_angle)
                            
                            # رسم motion arrow (أزرق) - smaller arrow
                            arrow_scale = 12.0
                            start = (int(centroid[0]), int(centroid[1]))
                            end = (int(centroid[0] + mean_u * arrow_scale), 
                                   int(centroid[1] + mean_v * arrow_scale))
                            cv2.arrowedLine(annotated_labeled_frame, start, end, 
                                          color=(255, 0, 0), thickness=2, tipLength=0.3)
                            
                        except Exception:
                            pass

                # Classification
                behavior = classify_behavior(slip_angle)
                color = get_behavior_color(behavior)
                
                # Update statistics
                stats = tracker_statistics[tracker_id]
                stats['total_frames'] += 1
                stats['current_behavior'] = behavior
                stats['behavior_counts'][behavior] += 1
                
                if slip_angle is not None:
                    stats['max_slip_angle'] = max(stats['max_slip_angle'], slip_angle)
                    history = list(slip_angles_history[tracker_id])
                    if history:
                        stats['avg_slip_angle'] = np.mean(history)

                # Draw heading orientation - smaller arrow
                if centroid is not None:
                    draw_centroid_and_orientation(annotated_labeled_frame, centroid, 
                                                 heading_angle, length=40, color=color)
                    
                    # Annotate slip angle and behavior - smaller text
                    if slip_angle is not None:
                        text_y = centroid[1] - 35
                        cv2.putText(annotated_labeled_frame, 
                                   f"Slip: {slip_angle:.1f}deg", 
                                   (centroid[0] - 50, text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                        
                        cv2.putText(annotated_labeled_frame, 
                                   behavior, 
                                   (centroid[0] - 50, text_y + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw dashboard
            draw_dashboard(annotated_labeled_frame, tracker_statistics, frame_number)
            
            sink.write_frame(frame=annotated_labeled_frame)
            prev_gray = gray.copy()

    # Save statistics to JSON
    if output_stats:
        stats_path = Path(target_video_path).with_suffix('.json')
        
        output_data = {
            'video_info': {
                'total_frames': video_info.total_frames,
                'fps': video_info.fps,
                'width': video_info.width,
                'height': video_info.height
            },
            'trackers': {}
        }
        
        for tracker_id, stats in tracker_statistics.items():
            output_data['trackers'][str(tracker_id)] = {
                'total_frames': stats['total_frames'],
                'max_slip_angle': float(stats['max_slip_angle']),
                'avg_slip_angle': float(stats['avg_slip_angle']),
                'behavior_distribution': dict(stats['behavior_counts']),
                'dominant_behavior': max(stats['behavior_counts'].items(), 
                                       key=lambda x: x[1])[0] if stats['behavior_counts'] else 'Unknown'
            }
        
        with open(stats_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✅ Statistics saved to: {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Drift and Zigzag Detection with Slip Angle Analysis (Cars Only)"
    )
    
    parser.add_argument(
        "--source_weights_path",
        required=True,
        help="Path to YOLO weights file",
        type=str,
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        required=True,
        help="Path to output video file",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Detection confidence threshold",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", 
        default=0.7, 
        help="IOU threshold for NMS", 
        type=float
    )
    parser.add_argument(
        "--output_stats",
        action="store_true",
        help="Save statistics to JSON file"
    )

    args = parser.parse_args()

    process_video(
        source_weights_path=args.source_weights_path,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        output_stats=args.output_stats,
    )