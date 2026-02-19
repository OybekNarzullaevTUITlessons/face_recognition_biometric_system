"""
Face Recognition Evaluation Script
-----------------------------------
• Dataset: dataset/<person_id>/*.jpg
• Embedding: FaceNet (InceptionResnetV1 - vggface2)
• Metric: Cosine similarity
• Output:
    - FAR/FRR table (CSV)
    - FAR/FRR plot (PNG)
    - ROC plot (PNG)
    - EER value
"""

import os
import glob
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import matplotlib.pyplot as plt


# =========================
# CONFIGURATION
# =========================
DATASET_DIR = "dataset"
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png")

MAX_GENUINE_PAIRS_PER_PERSON = 200
MAX_IMPOSTOR_PAIRS = 5000

THRESHOLDS = np.linspace(0.1, 0.95, 80)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================
# DATA STRUCTURE
# =========================
@dataclass
class FaceSample:
    person_id: str
    path: str
    emb: np.ndarray


# =========================
# UTIL FUNCTIONS
# =========================
def l2_normalize(x: np.ndarray, eps=1e-10):
    return x / (np.linalg.norm(x) + eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray):
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


def list_images(dataset_dir: str) -> Dict[str, List[str]]:
    persons = {}
    for person_id in sorted(os.listdir(dataset_dir)):
        pdir = os.path.join(dataset_dir, person_id)
        if not os.path.isdir(pdir):
            continue

        files = []
        for ext in IMG_EXTS:
            files.extend(glob.glob(os.path.join(pdir, ext)))

        if len(files) >= 2:
            persons[person_id] = sorted(files)

    return persons


def read_image(path: str):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Image read failed: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# =========================
# MODEL LOADING
# =========================
def build_models(device):
    mtcnn = MTCNN(
        image_size=160,
        margin=14,
        device=device
    )
    resnet = InceptionResnetV1(
        pretrained="vggface2"
    ).eval().to(device)

    return mtcnn, resnet


@torch.no_grad()
def extract_embedding(mtcnn, resnet, img_rgb, device):
    face_tensor = mtcnn(img_rgb)
    if face_tensor is None:
        return None

    face_tensor = face_tensor.unsqueeze(0).to(device)
    emb = resnet(face_tensor).cpu().numpy().reshape(-1)
    return l2_normalize(emb)


# =========================
# EMBEDDING CREATION
# =========================
def build_samples(dataset_dir, device):
    persons = list_images(dataset_dir)
    if not persons:
        raise RuntimeError("Dataset topilmadi yoki yetarli rasm yo‘q.")

    mtcnn, resnet = build_models(device)

    samples = []
    for person_id, paths in persons.items():
        for path in paths:
            try:
                img = read_image(path)
                emb = extract_embedding(mtcnn, resnet, img, device)
                if emb is not None:
                    samples.append(FaceSample(person_id, path, emb))
            except Exception as e:
                print(f"[WARN] {path}: {e}")

    return samples


# =========================
# PAIR GENERATION
# =========================
def build_pairs(samples):
    by_person = {}
    for i, s in enumerate(samples):
        by_person.setdefault(s.person_id, []).append(i)

    genuine_pairs = []
    for pid, idxs in by_person.items():
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                genuine_pairs.append((idxs[i], idxs[j]))

    impostor_pairs = []
    all_idx = list(range(len(samples)))

    while len(impostor_pairs) < MAX_IMPOSTOR_PAIRS:
        a, b = random.sample(all_idx, 2)
        if samples[a].person_id != samples[b].person_id:
            impostor_pairs.append((a, b))

    return genuine_pairs[:MAX_GENUINE_PAIRS_PER_PERSON], impostor_pairs


# =========================
# SCORE COMPUTATION
# =========================
def compute_scores(samples, pairs):
    scores = []
    for i, j in pairs:
        scores.append(cosine_similarity(samples[i].emb, samples[j].emb))
    return np.array(scores)


# =========================
# FAR / FRR
# =========================
def compute_far_frr(genuine_scores, impostor_scores):
    far_list = []
    frr_list = []

    for t in THRESHOLDS:
        far = np.mean(impostor_scores >= t)
        frr = np.mean(genuine_scores < t)
        far_list.append(far)
        frr_list.append(frr)

    return np.array(far_list), np.array(frr_list)


def find_eer(far, frr):
    diff = np.abs(far - frr)
    idx = np.argmin(diff)
    return THRESHOLDS[idx], (far[idx] + frr[idx]) / 2


# =========================
# MAIN PIPELINE
# =========================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    samples = build_samples(DATASET_DIR, device)
    print(f"[INFO] Total samples: {len(samples)}")

    genuine_pairs, impostor_pairs = build_pairs(samples)
    print(f"[INFO] Genuine pairs: {len(genuine_pairs)}")
    print(f"[INFO] Impostor pairs: {len(impostor_pairs)}")

    genuine_scores = compute_scores(samples, genuine_pairs)
    impostor_scores = compute_scores(samples, impostor_pairs)

    far, frr = compute_far_frr(genuine_scores, impostor_scores)
    eer_thr, eer_val = find_eer(far, frr)

    print("\n===== RESULTS =====")
    print(f"EER threshold : {eer_thr:.4f}")
    print(f"EER value     : {eer_val*100:.2f}%")

    # Save table
    df = pd.DataFrame({
        "threshold": THRESHOLDS,
        "FAR": far,
        "FRR": frr
    })
    df.to_csv("far_frr_table.csv", index=False)

    # FAR/FRR plot
    plt.figure()
    plt.plot(THRESHOLDS, far, label="FAR")
    plt.plot(THRESHOLDS, frr, label="FRR")
    plt.axvline(eer_thr, linestyle="--", label="EER")
    plt.legend()
    plt.grid()
    plt.savefig("far_frr_plot.png")

    # ROC
    plt.figure()
    plt.plot(far, 1 - frr)
    plt.xlabel("FAR")
    plt.ylabel("TPR (1-FRR)")
    plt.title("ROC Curve")
    plt.grid()
    plt.savefig("roc_curve.png")

    print("[INFO] Files saved:")
    print(" - far_frr_table.csv")
    print(" - far_frr_plot.png")
    print(" - roc_curve.png")


if __name__ == "__main__":
    main()
