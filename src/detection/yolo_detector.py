import cv2
import time
import yaml
from ultralytics import YOLO

def run_detection():
    # Load configuration
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    video_source = config['camera']['rtsp_url']
    conf_thresh = config['model']['confidence_threshold']
    target_classes = config['model']['classes']

    # Load YOLOv8 Nano model (downloads automatically on first run)
    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error opening video source: {video_source}")
        return

    frame_count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video stream.")
                break
              
            frame_count += 1
            start_time = time.time()

            # Run inference filtering by confidence and target classes
            results = model(frame, conf=conf_thresh, classes=target_classes, verbose=False)

            infer_time = (time.time() - start_time) * 1000 # in ms
            fps = 1000 / infer_time if infer_time > 0 else 0

            # Count total detected objects in this frame
            detections = len(results[0].boxes)

            if frame_count % 30 == 0:
                print(f"[Frame {frame_count}] Detections: {detections} | Inference Latency: {infer_time:.2f} ms | FPS: {fps:.2f}")

    except KeyboardInterrupt:
        print("\nStopping detection pipeline...")

    cap.release()

if __name__ == "__main__":
    run_detection()
