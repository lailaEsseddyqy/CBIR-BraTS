# src/models/brats_classifier.py
"""
Classifieur de grade tumoral sur embeddings SupCon figés (256D).

Usage futur — Classification-Guided Retrieval :
  1. encode(image)  → vecteur SupCon pour Qdrant (similarité visuelle)
  2. predict_grade(image) → grade prédit pour filtrer les résultats (réduction semantic gap)
"""
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
from typing import Optional, Union

from dotenv import load_dotenv

from src.models.autoencoder_supervised import BraTSAutoencoderSupervised
from src.training.grade_constants import GRADE_NAMES, IDX_TO_GRADE, NUM_GRADES


def resolve_supcon_ckpt_path(saved_path: str = "") -> str:
    """
    Retrouve le checkpoint SupCon : chemin enregistré dans le .ckpt classifieur,
    puis CHECKPOINT_PATH_SUPCON (.env), puis emplacements courants du projet.
    """
    load_dotenv()
    candidates = []
    if saved_path:
        candidates.append(saved_path)
        basename = os.path.basename(saved_path)
        candidates.append(os.path.join("saved_models", basename))
    env_path = os.getenv("CHECKPOINT_PATH_SUPCON", "")
    if env_path:
        candidates.append(env_path)
    candidates.extend([
        "./saved_models/model_final_v2.ckpt",
        "./saved_models/brats_supcon_best.ckpt",
    ])
    seen = set()
    for raw in candidates:
        if not raw:
            continue
        path = os.path.normpath(os.path.abspath(raw))
        if path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path):
            return path

    tried = [p for p in candidates if p]
    raise FileNotFoundError(
        "Checkpoint SupCon introuvable pour le classifieur CGR.\n"
        f"  Chemins testés : {tried}\n"
        "  Définissez CHECKPOINT_PATH_SUPCON dans .env"
    )


def _inference_state_dict(full: dict) -> dict:
    """Exclut les clés entraînement-only (loss pondérée, métriques Lightning)."""
    skip_prefixes = ("criterion.", "train_acc.", "val_acc.")
    return {k: v for k, v in full.items() if not k.startswith(skip_prefixes)}


class BraTSClassifierGuided(pl.LightningModule):
    """
    Tête de classification légère au-dessus de l'encodeur SupCon gelé.
    Seuls les poids du MLP `classifier` sont entraînés.
    """

    GRADE_NAMES = GRADE_NAMES

    def __init__(
        self,
        supcon_ckpt_path: str,
        num_classes: int = NUM_GRADES,
        lr: float = 1e-3,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        supcon_ckpt_path = resolve_supcon_ckpt_path(supcon_ckpt_path)
        self.supcon = BraTSAutoencoderSupervised.load_from_checkpoint(supcon_ckpt_path)
        self.supcon.eval()
        for param in self.supcon.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        weight = class_weights.float() if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=weight)

        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc   = MulticlassAccuracy(num_classes=num_classes)

    # ── Forward ───────────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Vecteur SupCon 256D (L2-normalisé) — pour indexation / recherche Qdrant."""
        self.supcon.eval()
        with torch.no_grad():
            _, embedding = self.supcon(x)
        return embedding

    def forward(self, x: torch.Tensor):
        embedding = self.encode(x).detach()
        logits = self.classifier(embedding)
        return embedding, logits

    @torch.no_grad()
    def predict_grade(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retourne (embedding, indice_classe, probabilités).
        Pour Classification-Guided Retrieval : filtrer Qdrant sur IDX_TO_GRADE[pred].
        """
        self.eval()
        embedding, logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        pred  = torch.argmax(probs, dim=1)
        return embedding, pred, probs

    @staticmethod
    def grade_name(class_idx: int) -> str:
        return IDX_TO_GRADE.get(int(class_idx), "Inconnu")

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, os.PathLike],
        map_location=None,
        **kwargs,
    ):
        """
        Charge le classifieur sans recharger SupCon depuis un chemin obsolete
        enregistré dans hyper_parameters (souvent relatif à la machine d'entrainement).
        """
        ckpt = torch.load(checkpoint_path, map_location=map_location or "cpu", weights_only=False)
        hp = dict(ckpt.get("hyper_parameters") or {})
        supcon_path = resolve_supcon_ckpt_path(hp.get("supcon_ckpt_path", ""))

        model = cls(
            supcon_ckpt_path=supcon_path,
            num_classes=hp.get("num_classes", NUM_GRADES),
            lr=hp.get("lr", 1e-3),
        )
        state_dict = _inference_state_dict(ckpt["state_dict"])
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    # ── Lightning steps ───────────────────────────────────────────────────

    def _shared_step(self, batch, stage: str):
        images, labels = batch
        _, logits = self(images)
        loss = self.criterion(logits, labels)

        acc_metric = self.train_acc if stage == "train" else self.val_acc
        acc_metric.update(logits, labels)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc",  acc_metric, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
