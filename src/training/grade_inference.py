# src/training/grade_inference.py
"""
Inférence du grade tumoral depuis les masques de segmentation BraTS 2021.
Logique identique à generate_fake_metadata.infer_grade (sans dépendance MongoDB/Qdrant).
"""
import os
from pathlib import Path

import numpy as np

BRATS_DIR = os.getenv("BRATS_DIR", "./Data/BraTS2021_Training_Data")


from typing import Optional


def find_seg_file(patient_id: str) -> Optional[str]:
    candidates = [
        Path(BRATS_DIR) / patient_id / f"{patient_id}_seg.nii.gz",
        Path(BRATS_DIR) / f"BraTS2021_{patient_id.split('_')[-1]}"
        / f"BraTS2021_{patient_id.split('_')[-1]}_seg.nii.gz",
        Path(BRATS_DIR) / patient_id / f"{patient_id}_seg.nii",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def infer_grade_from_segmentation(patient_id: str) -> str:
    """
    Règles BraTS 2021 (labels observés : 0, 1, 2, 4) :
      ET(4) + NCR(1) → Grade IV
      ET(4) seul     → Grade III
      NCR/ED sans ET → Grade II
    """
    try:
        import nibabel as nib
    except ImportError:
        return "Inconnu"

    seg_path = find_seg_file(patient_id)
    if seg_path is None:
        return "Inconnu"

    try:
        seg = nib.load(seg_path).get_fdata().astype(np.uint8)
    except Exception:
        return "Inconnu"

    has_ncr = np.any(seg == 1)
    has_ed  = np.any(seg == 2)
    has_et  = np.any(seg == 4)

    if has_et and has_ncr:
        return "Grade IV"
    if has_et and not has_ncr:
        return "Grade III"
    if (has_ncr or has_ed) and not has_et:
        return "Grade II"
    return "Inconnu"
