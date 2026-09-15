import os
import cv2
import numpy as np

class FaceRecognizer:
    def _init_(self, known_faces_dir="known_faces", tolerance=0.55):
        self.known_face_encodings = []
        self.known_face_names = []
        self.tolerance = tolerance
        
        # Disable dlib CUDA initialization globally via C++ runtime hook
        try:
            import dlib
            dlib.DLIB_USE_CUDA = False
        except Exception:
            pass

        import face_recognition
        self.fr = face_recognition
        self.load_known_faces(known_faces_dir)

    def load_known_faces(self, known_faces_dir):
        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)
            return

        print("Loading known faces...")
        for filename in os.listdir(known_faces_dir):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(known_faces_dir, filename)
                image = self.fr.load_image_file(path)

                # Generate encodings
                encodings = self.fr.face_encodings(image)

                if len(encodings) > 0:
                    self.known_face_encodings.append(encodings[0])
                    name = os.path.splitext(filename)[0].replace("_", " ").title()
                    self.known_face_names.append(name)
                    print(f"Loaded profile: {name}")
                else:
                    print(f"Warning: No face found in {filename}")

    def recognize_faces(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = self.fr.face_locations(rgb_frame, model="hog")
        face_encodings = self.fr.face_encodings(rgb_frame, face_locations)

        results = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            if self.known_face_encodings:
                matches = self.fr.compare_faces(
                    self.known_face_encodings, face_encoding, tolerance=self.tolerance
                )
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]

            results.append({"name": name, "box": (left, top, right, bottom)})

        return results
