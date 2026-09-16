import os
import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

class FaceRecognizer:
    def __init__(self, known_faces_dir="known_faces", tolerance=0.6):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # MTCNN for face detection, InceptionResnetV1 for 512-d embeddings
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        self.known_face_embeddings = []
        self.known_face_names = []
        self.tolerance = tolerance

        self.load_known_faces(known_faces_dir)

    def load_known_faces(self, known_faces_dir):
        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)
            return

        print("Loading known faces via PyTorch...")
        for filename in os.listdir(known_faces_dir):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(known_faces_dir, filename)
                img = Image.open(path).convert('RGB')

                # Detect face and calculate embedding
                faces = self.mtcnn(img)
                if faces is not None:
                  # Calculate 512-dim embedding
                    embedding = self.resnet(faces[0].unsqueeze(0).to(self.device))
                    self.known_face_embeddings.append(embedding.detach())
                    name = os.path.splitext(filename)[0].replace("_", " ").title()
                    self.known_face_names.append(name)
                    print(f"Loaded profile: {name}")
                else:
                    print(f"Warning: No face found in {filename}")

    def recognize_faces(self, frame):
        # Convert OpenCV BGR to PIL RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)

        # Detect face bounding boxes and cropped faces
        boxes, _ = self.mtcnn.detect(img)
        faces = self.mtcnn(img)

        results = []
        if boxes is not None and faces is not None:
            faces = faces.to(self.device)
            embeddings = self.resnet(faces).detach()

            for box, embedding in zip(boxes, embeddings):
                name = "Unknown"
                if self.known_face_embeddings:
                    # Calculate Euclidean distance against stored vectors
                    distances = [torch.dist(embedding, known).item() for known in self.known_face_embeddings]
                    min_dist = min(distances)

                    if min_dist < self.tolerance:
                        idx = distances.index(min_dist)
                        name = self.known_face_names[idx]
                      
                  left, top, right, bottom = map(int, box)
                results.append({"name": name, "box": (left, top, right, bottom)})

        return results
                          
