import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

DATASET_DIR = "dataset"
THRESHOLD = 0.90

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {device}")

mtcnn = MTCNN(image_size=160, margin=20, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)


def l2_normalize(x):
    return x / (np.linalg.norm(x) + 1e-10)


def cosine_similarity(a, b):
    return float(np.dot(a, b))


# ==========================
# 1. DATABASE YUKLASH
# ==========================
print("[INFO] Embeddings yuklanmoqda...")
database = {}

for person in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(person_path):
        continue

    embeddings = []

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = mtcnn(rgb)

        if face is not None:
            face = face.unsqueeze(0).to(device)
            with torch.no_grad():
                emb = resnet(face).cpu().numpy().flatten()
                emb = l2_normalize(emb)
                embeddings.append(emb)

    if embeddings:
        database[person] = l2_normalize(np.mean(embeddings, axis=0))

print(f"[INFO] {len(database)} ta foydalanuvchi yuklandi")

if not database:
    print("Dataset bo‘sh!")
    exit()


# ==========================
# 2. REAL-TIME TANISH
# ==========================
cap = cv2.VideoCapture(0)
print("[INFO] Q bilan chiqish mumkin")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # To‘g‘ridan-to‘g‘ri aligned face olish
    face_tensor, prob = mtcnn(rgb, return_prob=True)

    name = "UNKNOWN"

    if face_tensor is not None and prob > 0.90:
        face_tensor = face_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            emb = resnet(face_tensor).cpu().numpy().flatten()
            emb = l2_normalize(emb)

        best_score = 0
        best_match = None

        for person, db_emb in database.items():
            score = cosine_similarity(emb, db_emb)
            if score > best_score:
                best_score = score
                best_match = person

        if best_score > THRESHOLD:
            name = f"WELCOME {best_match}"

    color = (0, 255, 0) if "WELCOME" in name else (0, 0, 255)

    cv2.putText(frame, name, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2)

    cv2.imshow("Real-Time Login Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
