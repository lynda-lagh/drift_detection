from ultralytics import YOLO
import cv2


def main():
    # Load model (use the local filename present in workspace)
    model = YOLO('yolo12n.pt')

    video_path = 'v1.mp4'
    results = model.predict(source=video_path, stream=True, conf=0.25)

    # Loop through video frames
    for r in results:
        frame = r.plot()  # YOLO draws boxes, masks, names ON the frame

        # Show the frame
        cv2.imshow("YOLO Video", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
