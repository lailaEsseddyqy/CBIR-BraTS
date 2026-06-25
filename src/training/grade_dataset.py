
"""
Dataset pour l'entraînement du classifieur de grade (Classification-Guided Retrieval).

Utilise :
  - brats_subset_5k/*.pt          — coupes pré-extraites
  - brats_subset_5k_cache.json    — alignement index → patient_id
  - brats_grade_labels.json       — labels int (0=II, 1=III, 2=IV), généré via grade_labels.py

Split train/val par patient_id (pas par coupe) pour éviter le data leakage.
"""
import json
import os
import random

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from src.training.grade_constants import NUM_GRADES
from src.training.grade_labels import export_grade_labels, patient_id_from_nifti_path


class BraTSGradeDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        cache_json: str,
        grades_json: str,
        is_train: bool = True,
        val_ratio: float = 0.2,
        split_seed: int = 42,
        auto_export_labels: bool = True,
    ):
        self.data_dir = data_dir
        self.is_train = is_train

        if auto_export_labels and (
            not os.path.exists(grades_json) or _labels_file_invalid(grades_json)
        ):
            print(f"[GradeDataset] Generation / regeneration des labels : {grades_json}")
            export_grade_labels(data_dir, cache_json, grades_json)

        all_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".pt"))
        with open(cache_json, encoding="utf-8") as f:
            cache = json.load(f)
        with open(grades_json, encoding="utf-8") as f:
            raw_labels = json.load(f)

        all_labels = _parse_raw_labels(raw_labels)

        if len(all_files) != len(all_labels) or len(all_files) != len(cache):
            raise ValueError(
                f"Mismatch fichiers/labels/cache : "
                f"{len(all_files)} .pt, {len(all_labels)} labels, {len(cache)} cache"
            )

        patient_ids = [
            patient_id_from_nifti_path(cache[i][0]) for i in range(len(cache))
        ]
        val_patients = _val_patient_set(patient_ids, val_ratio, split_seed)

        self.files, self.labels = [], []
        for fname, label, pid in zip(all_files, all_labels, patient_ids):
            if label == -1:
                continue
            in_val = pid in val_patients
            if is_train and in_val:
                continue
            if not is_train and not in_val:
                continue
            self.files.append(fname)
            self.labels.append(label)

        split_name = "train" if is_train else "val"
        print(
            f"[GradeDataset] {split_name} : {len(self.labels)} coupes "
            f"(patients val={len(val_patients)}) — augmentation={is_train}"
        )
        for c in range(NUM_GRADES):
            print(f"  Classe {c} : {self.labels.count(c)} coupes")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.files[idx])
        img = torch.load(path, weights_only=True).float()
        label = self.labels[idx]

        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)

        if self.is_train:
            if random.random() > 0.5:
                img = TF.hflip(img)
            if random.random() > 0.5:
                img = TF.rotate(img, random.uniform(-15, 15))
            if random.random() > 0.5:
                img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))
            if random.random() > 0.5:
                img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))

        return img, label

    def get_labels(self) -> list:
        return self.labels


def _labels_file_invalid(grades_json: str) -> bool:
    try:
        with open(grades_json, encoding="utf-8") as f:
            raw = json.load(f)
        parsed = _parse_raw_labels(raw)
        return sum(1 for l in parsed if l != -1) == 0
    except Exception:
        return True


def _parse_raw_labels(raw_labels: list) -> list[int]:
    parsed = []
    for entry in raw_labels:
        if isinstance(entry, list):
            if any(isinstance(x, str) for x in entry):
                nums = [x for x in entry if isinstance(x, (int, float))]
                parsed.append(int(nums[0]) if nums else -1)
            elif len(entry) > 1:
                parsed.append(entry.index(max(entry)))
            else:
                parsed.append(int(entry[0]))
        else:
            parsed.append(int(entry))
    return parsed


def _val_patient_set(patient_ids: list[str], val_ratio: float, seed: int) -> set[str]:
    unique = sorted(set(patient_ids))
    rng = random.Random(seed)
    rng.shuffle(unique)
    n_val = max(1, int(len(unique) * val_ratio))
    return set(unique[:n_val])
