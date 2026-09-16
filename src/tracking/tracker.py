import cv2
import time
import yaml
from ultralytics import YOLO

def run_tracking():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(config['camera']['rtsp_url'])

    frame_count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            start_time = time.time()

            # Run YOLO with built-in ByteTrack tracker
            results = model.track(frame, persist=True, conf=config['model']['confidence_threshold'],
                                  classes=config['model']['classes'], verbose=False)

            infer_time = (time.time() - start_time) * 1000

            # Extract tracking IDs if present
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                if frame_count % 30 == 0:
                    print(f"[Frame {frame_count}] Tracking IDs Active: {track_ids} | Latency: {infer_time:.2f} ms")
                else:
                if frame_count % 30 == 0:
                    print(f"[Frame {frame_count}] Detections: {len(results[0].boxes)} (No Active Track IDs)")

    except KeyboardInterrupt:
        print("\nStopping tracker...")

    cap.release()

if __name__ == "__main__":
    run_tracking()
