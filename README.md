# Face Recognition Biometric System (Python 3.11)

Bu loyiha biometrik tizimlar fani uchun ishlab chiqilgan. Tizim FaceNet
asosida yuz orqali tanib olishni amalga oshiradi va FAR/FRR/EER
metrikalarini hisoblaydi.

---

## 📌 Loyihaning imkoniyatlari

- 📷 Kamera orqali dataset yig'ish
- 🧠 FaceNet (facenet-pytorch) orqali embedding olish
- 🔍 Cosine similarity asosida taqqoslash
- 📊 FAR (False Accept Rate) hisoblash
- 📊 FRR (False Reject Rate) hisoblash
- 🎯 EER (Equal Error Rate) aniqlash
- 📈 Grafik va CSV natijalar saqlash

---

## 🛠 Texnologiyalar

- Python 3.11
- PyTorch
- facenet-pytorch
- OpenCV
- NumPy
- Pandas
- Matplotlib

---

## ⚙️ O'rnatish

Virtual environment yaratish:

```shell
python -m venv venv
.\venv\Scripts\activate
```

Paketlarni o'rnatish:

```shell
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install facenet-pytorch opencv-python numpy pandas matplotlib tqdm
```

---

## 📷 1. Dataset yig'ish

```shell
python collect_faces.py
```

Har bir foydalanuvchi uchun alohida papka yaratiladi:

dataset/ ali/ vali/

---

## 🧠 2. Baholash (FAR/FRR/EER)

```shell
python face_eval.py
```

Natijalar:

- far_frr_table.csv
- far_frr_plot.png
- Terminalda EER qiymati

---

## 📊 Baholash metrikalari

- FAR -- Begona foydalanuvchi o'tib ketish ehtimoli
- FRR -- Haqiqiy foydalanuvchi rad etilish ehtimoli
- EER -- FAR va FRR tenglashgan nuqta

---

## 🎯 Amaliy maqsad

- Biometrik tizimlar fanida laboratoriya ishi
- Threshold tanlashni o'rganish
- Xavfsizlik va qulaylik balansini aniqlash

---

## ⚠️ Talablar

- Kamida 2 ta foydalanuvchi
- Har birida kamida 2--5 ta rasm
- Yuz aniq va frontal bo'lishi kerak

---
