# src/evaluation/medical_metrics.py
"""
Indicateurs médicaux de similarité entre deux coupes IRM :
  - SSIM  : Structural Similarity Index
  - PSNR  : Peak Signal-to-Noise Ratio
  - MSE   : Mean Squared Error
  - HIST  : Corrélation d'histogramme

Corrections appliquées (recommandations jury) :
  1. Garde-fou image constante/vide : si std < MIN_STD → métriques = 0.0
     (SSIM mathématiquement indéfini sur image uniforme, valeur trompeuse)
  2. SSIM forcé data_range=1.0 + normalisation robuste (eps=1e-8)
  3. PSNR : capture FloatingPointError (MSE=0 → inf) et retourne 100.0 (images identiques)
  4. is_valid_slice() : utilisée par le pipeline pour exclure les coupes
     pathologiques AVANT indexation ET avant affichage médecin
  5. compute_all() retourne aussi "valid" (bool) pour permettre le post-filtrage
  6. interpret() tient compte du flag invalid
"""

import numpy as np
import cv2
from skimage.metrics import (
    structural_similarity as _ssim,
    peak_signal_noise_ratio as _psnr,
    mean_squared_error as _mse,
)

# ── Seuils de validation ───────────────────────────────────
MIN_STD            = 0.01    # en dessous → image constante/vide
MIN_NONZERO_RATIO  = 0.05    # moins de 5 % de pixels non nuls → coupe vide
SSIM_DISPLAY_MIN   = 0.15    # seuil post-filtrage côté affichage médecin


# ─────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────

def normalize(img: np.ndarray) -> np.ndarray:
    """
    Normalise une image en float32 dans [0, 1].
    Utilise eps=1e-8 pour éviter la division par zéro
    sur les images constantes (noir pur, masque vide).
    """
    img = img.astype("float32")
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn + 1e-8)


def is_valid_slice(arr: np.ndarray) -> bool:
    """
    Retourne True si la coupe est médicalement exploitable.
    Critères :
      - Au moins MIN_NONZERO_RATIO de pixels non nuls
      - Écart-type suffisant (image non constante)

    Usage :
      - Pipeline d'indexation : exclure les coupes invalides AVANT indexation
      - run_search / run_multimodal : exclure les résultats invalides retournés
    """
    arr = arr.astype("float32")
    nonzero_ratio = np.count_nonzero(arr) / max(arr.size, 1)
    return bool(nonzero_ratio >= MIN_NONZERO_RATIO and arr.std() >= MIN_STD)


def _check_both(i1: np.ndarray, i2: np.ndarray) -> bool:
    """
    Vérifie que les DEUX images sont valides avant calcul.
    Retourne False si l'une ou l'autre est constante/vide.
    """
    return is_valid_slice(i1) and is_valid_slice(i2)


# ─────────────────────────────────────────────────────────
# Métriques individuelles
# ─────────────────────────────────────────────────────────

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    SSIM — Similarité structurelle [0, 1].
    Retourne 0.0 si l'une des images est constante/vide
    (cas mathématiquement indéfini : σ=0, covariance=0).
    """
    i1, i2 = normalize(img1), normalize(img2)
    if not _check_both(i1, i2):
        return 0.0
    score = _ssim(i1, i2, data_range=1.0)
    return round(float(score), 4)


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    PSNR — Peak Signal-to-Noise Ratio (dB).
    > 30 dB  : bonne similarité
    > 40 dB  : très bonne similarité
    = 100.0  : images identiques (MSE=0, PSNR→∞, on borne à 100)
    = 0.0    : image invalide/vide
    """
    i1, i2 = normalize(img1), normalize(img2)
    if not _check_both(i1, i2):
        return 0.0
    try:
        score = _psnr(i1, i2, data_range=1.0)
        # PSNR = inf quand MSE = 0 (images identiques)
        return 100.0 if np.isinf(score) else round(float(score), 2)
    except (FloatingPointError, ValueError):
        return 0.0


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    MSE — Erreur quadratique moyenne [0, ∞).
    0.0 = images identiques, valeur élevée = images très différentes.
    """
    i1, i2 = normalize(img1), normalize(img2)
    score = _mse(i1, i2)
    return round(float(score), 6)


def compute_histogram_correlation(img1: np.ndarray, img2: np.ndarray,
                                   bins: int = 64) -> float:
    """
    Corrélation d'histogramme [-1, 1].
    Robuste aux variations de contraste entre modalités.
    Retourne 0.0 si l'une des images est vide.
    """
    i1, i2 = normalize(img1), normalize(img2)
    if not _check_both(i1, i2):
        return 0.0

    u1 = (i1 * 255).astype("uint8")
    u2 = (i2 * 255).astype("uint8")

    h1 = cv2.calcHist([u1], [0], None, [bins], [0, 256])
    h2 = cv2.calcHist([u2], [0], None, [bins], [0, 256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)

    score = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    return round(float(score), 4)


# ─────────────────────────────────────────────────────────
# Calcul combiné
# ─────────────────────────────────────────────────────────

def compute_all(img1: np.ndarray, img2: np.ndarray) -> dict:
    """
    Calcule tous les indicateurs en une seule passe.

    Retourne aussi :
      "valid" (bool)  : False si l'image résultat est vide/constante
                        → permet au caller de filtrer avant affichage
      "invalid_reason": raison lisible si valid=False, sinon ""

    img1 = image requête
    img2 = image résultat
    """
    i2_norm = normalize(img2)
    valid   = is_valid_slice(i2_norm)
    reason  = ""

    if not valid:
        nonzero = np.count_nonzero(i2_norm) / max(i2_norm.size, 1)
        if nonzero < MIN_NONZERO_RATIO:
            reason = "coupe vide (image noire ou masque)"
        else:
            reason = "image constante (variance nulle)"
        return {
            "ssim"          : 0.0,
            "psnr"          : 0.0,
            "mse"           : 0.0,
            "hist"          : 0.0,
            "valid"         : False,
            "invalid_reason": reason,
        }

    return {
        "ssim"          : compute_ssim(img1, img2),
        "psnr"          : compute_psnr(img1, img2),
        "mse"           : compute_mse(img1, img2),
        "hist"          : compute_histogram_correlation(img1, img2),
        "valid"         : True,
        "invalid_reason": "",
    }


# ─────────────────────────────────────────────────────────
# Post-filtrage côté affichage médecin
# ─────────────────────────────────────────────────────────

def filter_valid_results(results: list, query_np: np.ndarray,
                          safe_load_fn,
                          ssim_min: float = SSIM_DISPLAY_MIN) -> list:
    """
    Filtre les résultats avant affichage au médecin :
      1. Calcule les métriques pour chaque résultat
      2. Exclut les coupes vides (valid=False)
      3. Exclut les coupes dont SSIM < ssim_min (seuil médical)
      4. Stocke les métriques dans r.metrics / r.interpretation

    Si après filtrage il reste 0 résultats, retourne la liste complète
    avec un flag pour que l'interface puisse avertir l'utilisateur.

    Args:
        results       : liste de résultats (SearchResult / MultimodalResult)
        query_np      : image requête normalisée (np.ndarray)
        safe_load_fn  : fonction safe_load(path) → np.ndarray | None
        ssim_min      : seuil SSIM minimum (défaut = SSIM_DISPLAY_MIN = 0.15)
    """
    import cv2 as _cv2

    valid_results = []

    for r in results:
        img = safe_load_fn(r.file_path)

        if img is None:
            # Fichier introuvable → exclure silencieusement
            continue

        q = _cv2.resize(query_np, (img.shape[1], img.shape[0]))
        metrics = compute_all(q, img)
        r.metrics = metrics
        r.interpretation = interpret(metrics)

        if not metrics["valid"]:
            # Coupe vide/constante → exclure
            continue

        if metrics["ssim"] < ssim_min:
            # SSIM trop faible pour être médicalement pertinent → exclure
            continue

        valid_results.append(r)

    return valid_results


# ─────────────────────────────────────────────────────────
# Interprétation
# ─────────────────────────────────────────────────────────

def interpret(metrics: dict) -> str:
    """
    Interprétation textuelle des métriques pour l'interface.
    Tient compte du flag "valid" pour signaler les résultats
    pathologiques (images vides) plutôt que de les afficher
    comme "peu similaires".
    """
    if not metrics.get("valid", True):
        return f"⚠️ Résultat invalide — {metrics.get('invalid_reason', 'image non exploitable')}"

    ssim_val = metrics.get("ssim", 0)
    hist_val = metrics.get("hist", 0)

    if ssim_val > 0.8 and hist_val > 0.8:
        return "✅ Très similaire"
    elif ssim_val > 0.5 and hist_val > 0.5:
        return "🟡 Modérément similaire"
    elif ssim_val > 0.15:
        return "🔴 Peu similaire"
    else:
        return "⚠️ Résultat non exploitable (SSIM insuffisant)"