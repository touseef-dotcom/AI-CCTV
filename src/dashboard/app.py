from flask import Flask, Response
import cv2
import yaml
from ultralytics import YOLO
import sys
import os

# Add root directory to sys.path so it can find src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.detection.face_engine import FaceRecognizer

app = Flask(__name__)

# Load configuration
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize YOLO model and Face Recognizer
model = YOLO("yolov8n.pt")
face_engine = FaceRecognizer(known_faces_dir="known_faces")

def generate_frames():
    raw_url = config['camera']['rtsp_url']
    source = int(raw_url) if str(raw_url).isdigit() else raw_url
    cap = cv2.VideoCapture(source)

    frame_counter = 0
    last_detected_faces = []        for face in last_detected_faces:
            left, top, right, bottom = face["box"]
            name = face["name"]
      
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(annotated_frame, (left, top), (right, bottom), color, 2)
            cv2.putText(
                annotated_frame, name, (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2
            )

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '<h2 style="text-align:center;">AI CCTV Stream</h2><img src="/video_feed" width="100%"/>'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
