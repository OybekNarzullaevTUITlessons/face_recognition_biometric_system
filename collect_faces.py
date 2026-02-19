# ====== KUTUBXONALAR ======
import os              # Fayl va papkalar bilan ishlash
import time            # Vaqtni o‘lchash uchun (1 sekund interval)
import cv2             # Kamera va tasvir bilan ishlash (OpenCV)
import numpy as np     # Matritsalar (bu yerda deyarli ishlatilmayapti)
import torch           # GPU/CPU tanlash uchun
from facenet_pytorch import MTCNN  # Yuzni aniqlash modeli

# ====== FOYDALANUVCHI PARAMETRLARI ======

# Foydalanuvchi ID sini so‘raydi (masalan: oybek, ali, user1)
PERSON_ID = input("Person ID (masalan: oybek): ")

# Har bir shaxs uchun maksimal saqlanadigan rasm soni
MAX_IMAGES = 30

# Dataset papkasi: dataset/oybek kabi yaratiladi
SAVE_DIR = os.path.join("dataset", PERSON_ID)

# Agar papka mavjud bo‘lmasa, yaratadi
os.makedirs(SAVE_DIR, exist_ok=True)


# ====== QURILMA TANLASH (CPU yoki GPU) ======

# Agar CUDA (GPU) mavjud bo‘lsa → GPU ishlatadi
# Aks holda → CPU ishlatiladi
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] Device: {device}")


# ====== MTCNN MODEL ======

# MTCNN — yuzni aniqlash modeli
# image_size=160 → chiqish tasvir hajmi (FaceNet standart)
# margin=20 → yuz atrofida qo‘shimcha bo‘sh joy
# device → CPU yoki GPU
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    device=device
)


# ====== KAMERA ISHGA TUSHIRISH ======

# 0 → asosiy webcam
cap = cv2.VideoCapture(0)

# Kamera ochilmagan bo‘lsa dastur to‘xtaydi
if not cap.isOpened():
    print("Camera ochilmadi!")
    exit()

print("[INFO] Q harfi bilan chiqish mumkin")
print("[INFO] Dataset yig'ilmoqda...")


# ====== BOSHLANG‘ICH PARAMETRLAR ======

count = 0               # Saqlangan rasmlar soni
last_save = time.time() # Oxirgi saqlangan vaqt


# ====== ASOSIY SIKL ======
while True:

    # Kameradan frame olish
    ret, frame = cap.read()

    # Agar frame olinmasa → siklni to‘xtatadi
    if not ret:
        break

    # OpenCV BGR format beradi → RGB ga o‘tkazamiz
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ====== YUZNI ANIQLASH ======
    boxes, _ = mtcnn.detect(rgb)

    # Agar yuz topilgan bo‘lsa
    if boxes is not None:

        # Har bir topilgan yuz uchun
        for box in boxes:

            # Bounding box koordinatalari
            x1, y1, x2, y2 = map(int, box)

            # Frame ichidan yuz qismini kesib olamiz (crop)
            face = frame[y1:y2, x1:x2]

            # Yuzni 160x160 formatga keltiramiz (FaceNet talabi)
            face_resized = cv2.resize(face, (160, 160))

            # ====== RASMNI SAQLASH ======
            # Har 1 sekundda 1 ta rasm saqlaydi
            if time.time() - last_save > 1 and count < MAX_IMAGES:

                # Saqlanadigan rasm nomi
                img_path = os.path.join(SAVE_DIR, f"{count+1}.jpg")

                # Rasmni diskka yozish
                cv2.imwrite(img_path, face_resized)

                print(f"[SAVED] {img_path}")

                count += 1
                last_save = time.time()

            # Ekranda yuz atrofida yashil to‘rtburchak chizish
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


    # ====== EKRANGA MA'LUMOT CHIQARISH ======

    cv2.putText(frame,
                f"Images: {count}/{MAX_IMAGES}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    # Oynada video ko‘rsatish
    cv2.imshow("Dataset Collector", frame)

    # Agar foydalanuvchi Q tugmasini bossа → chiqish
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Agar kerakli rasm soni to‘plansa → to‘xtaydi
    if count >= MAX_IMAGES:
        break


# ====== YAKUNLASH ======

cap.release()          # Kamerani yopish
cv2.destroyAllWindows()  # Barcha oynalarni yopish

print("[INFO] Dataset yig'ish tugadi!")
