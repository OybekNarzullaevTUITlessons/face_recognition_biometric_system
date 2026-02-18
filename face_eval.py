import os
import glob
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1


# =========================
# CONFIG
# =========================
DATASET_DIR = "dataset"
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.webp")

# Genuine juftliklar sonini cheklash (katta datasetda tezlashadi)
MAX_GENUINE_PAIRS_PER_PERSON = 200
# Impostor juftliklar soni (umumiy)
MAX_IMPOSTOR_PAIRS = 5000

# Thresholdlar (cosine similarity)
THRESHOLDS = np.linspace(0.1, 0.95, 60)  # kerak bo'lsa kengaytiring

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


@dataclass
class FaceSample:
    person_id: str
    path: str
    emb: np.ndarray  # (512,)


def list_images(dataset_dir: str) -> Dict[str, List[str]]:
    persons = {}
    for person_id in sorted(os.listdir(dataset_dir)):
        pdir = os.path.join(dataset_dir, person_id)
        if not os.path.isdir(pdir):
            continue
        files = []
        for ext in IMG_EXTS:
            files.extend(glob.glob(os.path.join(pdir, ext)))
        files = sorted(files)
        if len(files) >= 2:  # EER uchun kamida 2 ta rasm yaxshi
            persons[person_id] = files
    return persons


def read_bgr(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Image read failed: {path}")
    return img


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def l2_normalize(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    n = np.linalg.norm(x)
    return x / (n + eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # a,b allaqachon L2-normalize bo'lsa ham, xavfsiz:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))


def build_models(device: str):
    mtcnn = MTCNN(
        image_size=160,
        margin=14,
        min_face_size=40,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        device=device
    )
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return mtcnn, resnet


@torch.no_grad()
def extract_embedding(mtcnn: MTCNN, resnet: InceptionResnetV1, img_rgb: np.ndarray, device: str) -> np.ndarray | None:
    # mtcnn PIL yoki numpy RGB qabul qiladi
    face_tensor = mtcnn(img_rgb)
    if face_tensor is None:
        return None
    face_tensor = face_tensor.unsqueeze(0).to(device)  # (1,3,160,160)
    emb = resnet(face_tensor).cpu().numpy().reshape(-1)  # (512,)
    emb = l2_normalize(emb)
    return emb


def make_samples(dataset_dir: str, device: str) -> List[FaceSample]:
    persons = list_images(dataset_dir)
    if not persons:
        raise RuntimeError(
            "Dataset topilmadi yoki har papkada kamida 2 rasm yo‘q.")

    mtcnn, resnet = build_models(device)
    samples: List[FaceSample] = []

    print(f"[INFO] Persons: {len(persons)}")
    for person_id, paths in persons.items():
        for path in paths:
            try:
                bgr = read_bgr(path)
                rgb = bgr_to_rgb(bgr)
                emb = extract_embedding(mtcnn, resnet, rgb, device)
                if emb is None:
                    # yuz topilmasa o'tkazib yuboramiz
                    continue
                samples.append(FaceSample(
                    person_id=person_id, path=path, emb=emb))
            except Exception as e:
                print(f"[WARN] {path}: {e}")

    # Har odamdan kamida 2 sample bo‘lishi kerak
    by_person = {}
    for s in samples:
        by_person.setdefault(s.person_id, []).append(s)
    by_person = {k: v for k, v in by_person.items() if len(v) >= 2}

    filtered = []
    for k, v in by_person.items():
        filtered.extend(v)

    print(
        f"[INFO] Samples kept: {len(filtered)} (persons kept: {len(by_person)})")
    if len(by_person) < 2:
        raise RuntimeError(
            "Kamida 2 ta odam bo‘lishi kerak (impostor pairlar uchun).")

    return filtered


def build_pairs(samples: List[FaceSample]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    # index bo'yicha juftliklar
    by_person: Dict[str, List[int]] = {}
    for i, s in enumerate(samples):
        by_person.setdefault(s.person_id, []).append(i)

    persons = sorted(by_person.keys())

    # Genuine: bir odam ichida
    genuine_pairs = []
    for pid in persons:
        idxs = by_person[pid]
        all_pairs = []
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                all_pairs.append((idxs[i], idxs[j]))
        random.shuffle(all_pairs)
        genuine_pairs.extend(all_pairs[:MAX_GENUINE_PAIRS_PER_PERSON])

    # Impostor: turli odamlar orasida
    impostor_pairs = []
    all_indices = list(range(len(samples)))
    # tez random sampling
    tries = 0
    while len(impostor_pairs) < MAX_IMPOSTOR_PAIRS and tries < MAX_IMPOSTOR_PAIRS * 20:
        a = random.choice(all_indices)
        b = random.choice(all_indices)
        if a == b:
            tries += 1
            continue
        if samples[a].person_id != samples[b].person_id:
            # tartib bir xil bo‘lib qolmasin
            if a > b:
                a, b = b, a
            impostor_pairs.append((a, b))
        tries += 1

    # unique qilib qo'yamiz
    impostor_pairs = list(set(impostor_pairs))
    random.shuffle(impostor_pairs)
    impostor_pairs = impostor_pairs[:MAX_IMPOSTOR_PAIRS]

    print(f"[INFO] Genuine pairs: {len(genuine_pairs)}")
    print(f"[INFO] Impostor pairs: {len(impostor_pairs)}")
    return genuine_pairs, impostor_pairs


def compute_scores(samples: List[FaceSample], pairs: List[Tuple[int, int]]) -> np.ndarray:
    scores = np.zeros(len(pairs), dtype=np.float32)
    for k, (i, j) in enumerate(pairs):
        scores[k] = cosine_similarity(samples[i].emb, samples[j].emb)
    return scores


def far_frr(scores_genuine: np.ndarray, scores_impostor: np.ndarray, thresholds: np.ndarray):
    # Accept = score >= threshold
    far_list = []
    frr_list = []

    for t in thresholds:
        # FAR: impostor accept / impostor total
        far = float(np.mean(scores_impostor >= t))
        # FRR: genuine reject / genuine total
        frr = float(np.mean(scores_genuine < t))
        far_list.append(far)
        frr_list.append(frr)

    return np.array(far_list), np.array(frr_list)


def find_eer(thresholds: np.ndarray, far: np.ndarray, frr: np.ndarray):
    diff = np.abs(far - frr)
    idx = int(np.argmin(diff))
    return float(thresholds[idx]), float((far[idx] + frr[idx]) / 2.0), idx


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    samples = make_samples(DATASET_DIR, device=device)
    genuine_pairs, impostor_pairs = build_pairs(samples)

    # Scorelar
    scores_genuine = compute_scores(samples, genuine_pairs)
    scores_impostor = compute_scores(samples, impostor_pairs)

    # FAR/FRR
    far, frr = far_frr(scores_genuine, scores_impostor, THRESHOLDS)

    # EER
    eer_thr, eer_val, eer_idx = find_eer(THRESHOLDS, far, frr)

    print("\n===== RESULTS =====")
    print(f"EER threshold: {eer_thr:.4f}")
    print(f"EER value    : {eer_val*100:.2f}%")
    print(f"FAR@EER      : {far[eer_idx]*100:.2f}%")
    print(f"FRR@EER      : {frr[eer_idx]*100:.2f}%")

    # Jadvalga chiqaramiz
    df = pd.DataFrame({
        "threshold": THRESHOLDS,
        "FAR": far,
        "FRR": frr,
        "abs_diff": np.abs(far - frr)
    }).sort_values("threshold")
    df.to_csv("far_frr_table.csv", index=False)
    print("[INFO] Saved: far_frr_table.csv")

    # Grafik
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(THRESHOLDS, far, label="FAR")
    plt.plot(THRESHOLDS, frr, label="FRR")
    plt.axvline(eer_thr, linestyle="--", label=f"EER thr={eer_thr:.3f}")
    plt.xlabel("Threshold (cosine similarity)")
    plt.ylabel("Rate")
    plt.title("FAR/FRR vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("far_frr_plot.png", dpi=150)
    print("[INFO] Saved: far_frr_plot.png")


if __name__ == "__main__":
    main()
