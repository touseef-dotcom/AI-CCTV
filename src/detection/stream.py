from detection.face_engine import FaceRecognizer

# Initialize the recognizer at application start
face_engine = FaceRecognizer(known_faces_dir="known_faces")

frame_counter = 0
last_detected_faces = []

def process_frame(frame):
    global frame_counter, last_detected_faces
    frame_counter += 1

    # Run heavy face recognition every 5th frame to protect Jetson FPS
    if frame_counter % 5 == 0:
        last_detected_faces = face_engine.recognize_faces(frame)

    # Draw recognized faces onto the current frame
    for face in last_detected_faces:
        left, top, right, bottom = face["box"]
        name = face["name"]

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(
            frame, name, (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2
        )
  
    return frame
