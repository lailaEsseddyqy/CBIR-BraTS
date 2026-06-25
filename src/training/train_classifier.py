# src/training/train_classifier.py
"""
Entraînement de la tête de classification pour Classification-Guided Retrieval.

Pipeline :
  SupCon (gelé, 256D) → MLP classifieur → grade OMS (II / III / IV)

Prérequis :
  1. Checkpoint SupCon : CHECKPOINT_PATH_SUPCON
  2. Labels de grade   : GRADES_LABELS_JSON (auto-généré si absent)

Usage :
  python src/training/train_classifier.py
"""
import os
import sys
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, WeightedRandomSampler
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.brats_classifier import BraTSClassifierGuided
from src.training.grade_constants import NUM_GRADES
from src.training.grade_dataset import BraTSGradeDataset

# ── Chemins (.env) ────────────────────────────────────────────────────────────
DATA_DIR          = os.getenv("DATA_DIR", "./Data/brats_subset_5k")
CACHE_JSON        = os.getenv("CACHE_JSON", "./Data/brats_subset_5k_cache.json")
GRADES_LABELS_JSON = os.getenv("GRADES_LABELS_JSON", "./Data/brats_grade_labels.json")
SUPCON_CKPT       = os.getenv("CHECKPOINT_PATH_SUPCON", "./saved_models/brats_supcon_best.ckpt")
OUTPUT_CKPT       = os.getenv("CLASSIFIER_CKPT", "./saved_models/brats_guided_classifier.ckpt")

BATCH_SIZE  = int(os.getenv("CLASSIFIER_BATCH_SIZE", 32))
MAX_EPOCHS  = int(os.getenv("CLASSIFIER_EPOCHS", 20))
NUM_WORKERS = 0  # obligatoire Windows


def _class_weights(labels: list[int]) -> torch.Tensor:
    """Poids inversement proportionnels à la fréquence (déséquilibre Grade IV ~94%)."""
    counts = torch.zeros(NUM_GRADES, dtype=torch.float32)
    for lbl in labels:
        counts[lbl] += 1
    counts = counts.clamp(min=1.0)
    weights = counts.sum() / (NUM_GRADES * counts)
    return weights


def _weighted_sampler(labels: list[int]) -> WeightedRandomSampler:
    class_w = _class_weights(labels)
    sample_w = [class_w[lbl].item() for lbl in labels]
    return WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)


def main():
    if not os.path.exists(SUPCON_CKPT):
        raise FileNotFoundError(
            f"Checkpoint SupCon introuvable : {SUPCON_CKPT}\n"
            "Définissez CHECKPOINT_PATH_SUPCON dans .env"
        )

    train_dataset = BraTSGradeDataset(
        data_dir=DATA_DIR,
        cache_json=CACHE_JSON,
        grades_json=GRADES_LABELS_JSON,
        is_train=True,
    )
    val_dataset = BraTSGradeDataset(
        data_dir=DATA_DIR,
        cache_json=CACHE_JSON,
        grades_json=GRADES_LABELS_JSON,
        is_train=False,
    )

    if len(train_dataset) == 0:
        raise RuntimeError("Dataset d'entraînement vide — vérifiez les labels de grade.")

    class_weights = _class_weights(train_dataset.get_labels())
    print(f"[train_classifier] Poids de classes : {class_weights.tolist()}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=_weighted_sampler(train_dataset.get_labels()),
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = BraTSClassifierGuided(
        supcon_ckpt_path=SUPCON_CKPT,
        num_classes=NUM_GRADES,
        class_weights=class_weights,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.dirname(OUTPUT_CKPT) or ".",
        filename="brats_guided_classifier",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=accelerator,
        callbacks=[checkpoint_cb, early_stop_cb],
        log_every_n_steps=10,
    )
    trainer.fit(model, train_loader, val_loader)

    best_path = checkpoint_cb.best_model_path or OUTPUT_CKPT
    print(f"\n[train_classifier] Meilleur checkpoint : {best_path}")
    print(
        "Prochaine étape — Classification-Guided Retrieval :\n"
        "  1. predict_grade(query) → grade prédit\n"
        "  2. Qdrant search + filtre MongoDB patient.grade == grade prédit"
    )


if __name__ == "__main__":
    main()
