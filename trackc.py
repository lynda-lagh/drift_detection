import argparse
from collections import defaultdict, deque

from tqdm import tqdm
from ultralytics import YOLO

import cv2
import numpy as np

import supervision as sv


def process_video(
    source_weights_path: str,
    source_video_path: str,
    target_video_path: str,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
) -> None:
    model = YOLO(source_weights_path)

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    # store recent anchor points for each tracker to draw trajectory lines
    coordinates = defaultdict(lambda: deque(maxlen=int(video_info.fps * 5)))
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)

    with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            results = model(
                frame, verbose=False, conf=confidence_threshold, iou=iou_threshold
            )[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)

            # get anchor points (bottom-center) in original frame coordinates
            try:
                anchors = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            except Exception:
                anchors = np.array([])

            if anchors.size:
                # anchors is expected as Nx2 array of (x, y)
                for tracker_id, (x, y) in zip(detections.tracker_id, anchors):
                    coordinates[tracker_id].append((int(x), int(y)))

            annotated_frame = box_annotator.annotate(
                scene=frame.copy(), detections=detections
            )

            annotated_labeled_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections
            )

            # draw trajectory lines for each tracker on the annotated frame
            for tracker_id, pts in coordinates.items():
                if len(pts) > 1:
                    pts_np = np.array(pts, dtype=np.int32)
                    cv2.polylines(annotated_labeled_frame, [pts_np], isClosed=False, color=(0, 255, 0), thickness=2)
                    # draw last position as filled circle
                    cv2.circle(annotated_labeled_frame, tuple(pts[-1]), radius=3, color=(0, 0, 255), thickness=-1)

            sink.write_frame(frame=annotated_labeled_frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video Processing with YOLO and ByteTrack"
    )
    
    parser.add_argument(
        "--source_weights_path",
        required=True,
        help="Path to the source weights file",
        type=str,
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        required=True,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )

    args = parser.parse_args()

    process_video(
        source_weights_path=args.source_weights_path,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
    )