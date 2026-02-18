import os
import time
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

# ====== CONFIG ======
PERSON_ID = input("Person ID (masalan: oybek): ")
MAX_IMAGES = 30
SAVE_DIR = os.path.join("dataset", PERSON_ID)
os.makedirs(SAVE_DIR, exist_ok=True)

# ====== DEVICE ======
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {device}")

# ====== MTCNN ======
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    device=device
)

# ====== CAMERA ======
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera ochilmadi!")
    exit()

print("[INFO] Q harfi bilan chiqish mumkin")
print("[INFO] Dataset yig'ilmoqda...")

count = 0
last_save = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face detect
    boxes, _ = mtcnn.detect(rgb)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]

            # yuzni 160x160 qilish
            face_resized = cv2.resize(face, (160, 160))

            # har 1 sekundda saqlash
            if time.time() - last_save > 1 and count < MAX_IMAGES:
                img_path = os.path.join(SAVE_DIR, f"{count+1}.jpg")
                cv2.imwrite(img_path, face_resized)
                print(f"[SAVED] {img_path}")
                count += 1
                last_save = time.time()

            # ekranda chizish
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(frame, f"Images: {count}/{MAX_IMAGES}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Dataset Collector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if count >= MAX_IMAGES:
        break

cap.release()
cv2.destroyAllWindows()

print("[INFO] Dataset yig'ish tugadi!")
