# CBIR-SYS — Mémoire Master (Projet de Recherche)

> **Objectif du document** : fournir une description structurée, technique et exploitable pour la rédaction d’un **mémoire de Master de recherche** sur le système **CBMIR** (Content-Based Medical Image Retrieval) développé dans ce projet.
>
> Le système vise à **retrouver des coupes IRM cérébrales similaires** à partir du **contenu visuel** de l’image, tout en intégrant des **contraintes cliniques** (ex. grade tumoral OMS) via un mécanisme de **Classification-Guided Retrieval (CGR)**.

---

## 1. Contexte et motivation

La recherche d’images médicales similaires est un besoin central en imagerie cérébrale (neuro-oncologie notamment). Les approches basées sur la similarité visuelle peuvent être insuffisantes car elles ignorent le contexte clinique. Dans le dataset **BraTS 2021**, l’un des défis majeurs est le **déséquilibre de classes** (notamment un fort pourcentage de Grade IV). Ce déséquilibre peut biaiser une métrique de type Precision@K calculée globalement.

Le projet propose donc :
1. Une représentation visuelle discriminante via des réseaux d’encodeurs.
2. Une base de recherche vectorielle (Qdrant) pour des requêtes rapides par similarité.
3. Une intégration clinique guidée par une **prédiction de grade** (MLP entraîné sur embeddings SupCon gelés), permettant de filtrer les résultats candidats.
4. Une évaluation **stratifiée** pour refléter une performance réaliste sur toutes les classes.

---

## 2. Vue d’ensemble du système

### 2.1. Trois modes de fonctionnement
Le cœur du système s’articule autour de trois configurations présentées dans `src/recherche/unified_engine.py` et `src/recherche/multimodal_search.py` :

- **Baseline (non supervisé)**
  - Encodage via un **Auto-encodeur BraTS** (`BraTSAutoencoderLightning`).
  - Indexation Qdrant sur une collection de vecteurs.

- **SupCon (contrastif supervisé)**
  - Encodage via un **Auto-encodeur supervisé** (`BraTSAutoencoderSupervised`).
  - Entraînement avec une **Supervised Contrastive Loss** (`SupConLoss`).
  - Indexation Qdrant sur une collection dédiée.

- **Guided / CGR (Classification-Guided Retrieval)**
  - Un **classifieur de grade** (`BraTSClassifierGuided`) prédit le grade à partir des embeddings SupCon.
  - Le moteur effectue ensuite une recherche vectorielle SupCon **filtrée** par le grade prédit.

### 2.2. Composants logiciels
Les éléments principaux observés dans le repo :

- **Moteurs de recherche**
  - `UnifiedSearchEngine` : recherche unifiée multi-modèles.
  - `MultimodalSearchEngine` : extension avec `PatientFilter` et requêtes multimodales (dans l’interface clinique).

- **Réseaux de caractéristiques**
  - `BraTSAutoencoderLightning` : baseline non supervisé.
  - `BraTSAutoencoderSupervised` : SupCon (supervised contrastive) avec projection/embedding.
  - `BraTSClassifierGuided` : classifieur MLP au-dessus du SupCon gelé.

- **Infrastructure de données**
  - **Qdrant** : stockage des vecteurs et recherche ANN (cosinus).
  - **MongoDB** : métadonnées des patients/coupes, récupération de `file_path` et attributs cliniques.

- **Évaluation et métriques médicales**
  - `Precision@K` stratifiée : `src/evaluation/precision_at_k.py`.
  - Similarité structurelle / signaux : SSIM, PSNR, MSE, histogram correlation dans `src/evaluation/medical_metrics.py`.

- **Interfaces Streamlit**
  - `src/interface/app_tech.py` : dashboard technique (pipeline explorer, comparaison modèle, évaluation P@K, architecture).
  - `src/interface/app_clinique.py` : interface clinique (filtres patient + choix CGR vs SupCon).

---

## 3. Prétraitement et représentation

### 3.1. Prétraitement de la requête
Dans les interfaces Streamlit (`app_tech.py`, `app_clinique.py`), la requête est manipulée comme une image 2D (coupe) :
- Conversion en **grayscale** (moyenne des canaux si image RGB).
- **Normalisation** dans la plage [0, 1] via min-max.
- Resize / interpolation bilinéaire vers **128×128**.
- Transformation en tenseur PyTorch avec forme attendue par les modèles.

> Ce point est utile pour la rédaction : le système ne manipule pas directement les volumes 3D, mais des coupes normalisées.

### 3.2. Encodeurs (baseline vs SupCon)

#### Baseline : auto-encodeur non supervisé
- Encodeur CNN 2D avec plusieurs couches convolutionnelles.
- Projection vers un vecteur latent via `embedding_layer`.
- Recherche dans Qdrant sur la collection baseline.

#### SupCon : auto-encodeur supervisé contrastif
- Encodeur plus profond (observé dans le code SupCon) menant à une projection **embedding 256D**.
- Le modèle renvoie :
  - la reconstruction (decoder)
  - l’embedding **L2-normalisé**
- La loss combine :
  - reconstruction (MSE) et
  - `SupConLoss` sur l’embedding, avec température.

---

## 4. Indexation vectorielle et gestion des métadonnées

### 4.1. Qdrant
Le module `src/db/connections.py` définit :
- URL Qdrant (local par défaut).
- Dimensions vectorielles (`LATENT_DIM` par défaut 256).
- Distance : **cosinus**.
- Collections :
  - baseline : `QDRANT_COLLECTION` (dans `.env`, défaut `brats_embeddings`).
  - supcon : observée comme `brats_supcon_embeddings` dans les moteurs.

### 4.2. MongoDB
- Collection Mongo : `slices_metadata` (dans `.env`).
- Rôle : stocker le mapping `slice_id → file_path + patient + grade + stats + modalite + slice_z`.
- Création index unique sur `slice_id`.

### 4.3. Id générés pour le filtrage
Pour le mode Guided, l’implémentation combine :
- prédiction du grade
- génération d’IDs Qdrant à partir de `slice_id` via `uuid.uuid5`
- application d’un filtre `HasIdCondition` sur Qdrant.

---

## 5. Classification-Guided Retrieval (CGR)

### 5.1. Intuition scientifique
L’idée de CGR consiste à réduire le **semantic gap** : au lieu de retourner les images visuellement proches uniquement, le système contraint la recherche aux cas cliniquement pertinents.

### 5.2. Architecture du classifieur
Le classifieur est `BraTSClassifierGuided` :
- Base : **SupCon gelé** (aucun gradient, `requires_grad = False`).
- Tête : MLP
  - Linear(256→128), BatchNorm1d, ReLU, Dropout(0.3), Linear(128→num_classes).
- Sortie :
  - prédiction `pred` via softmax
  - probas cliniques (utilisées dans l’interface clinique).

### 5.3. Entraînement du classifieur de grade
La procédure d’entraînement se trouve dans `src/training/train_classifier.py` :
- Dataset `BraTSGradeDataset` (split train/val par **patient_id**).
- Gestion du déséquilibre via **WeightedRandomSampler** et poids de classes.
- Loss : `CrossEntropyLoss` pondérée.
- Entraînement PyTorch Lightning avec callbacks :
  - `ModelCheckpoint` monitor `val_acc`
  - `EarlyStopping` monitor `val_loss`.

---

## 6. Évaluation scientifique : Precision@K stratifiée

### 6.1. Problème du déséquilibre
Le fichier `src/evaluation/precision_at_k.py` explicite que Grade IV représente une large proportion (≈ 94%), ce qui peut pousser une approche globale vers une performance proche du hasard.

### 6.2. Stratégies de métriques disponibles
Le code met en avant :
1. **Precision@K STRATIFIÉE** (équilibrage par grade)
   - sélection de `N_PER_GRADE` requêtes **par grade**.
2. **Balanced Precision@K**
   - macro-averaging de P@K par grade.
3. **Precision@K GLOBALE**
   - version standard mais signalée comme potentiellement biaisée.

Le repère du hasard équilibré est donné comme **1/3 ≈ 0.33** (trois classes : II/III/IV).

### 6.3. Sélection stratifiée des requêtes
La sélection est réalisée par `pick_query_slices_stratified(...)` :
- construit patient_id → grade + indices de coupes.
- sélectionne `n_per_grade` patients par grade.
- choisit une coupe “médiane en Z” pour maximiser l’information.

### 6.4. Gestion des erreurs Qdrant
La fonction `_search_with_retry` ajoute :
- retry exponentiel en cas d’erreurs Qdrant.
- délai entre requêtes (évite le rate-limiting).

---

## 7. Similarité médicale : SSIM / PSNR / MSE / histogram correlation

### 7.1. Pourquoi ces métriques ?
Même si la recherche se fait dans l’espace vectoriel, l’évaluation “médicale” et le filtrage affiché au médecin reposent sur des métriques de similarité **image-based**.

### 7.2. Garde-fous (anti faux positifs)
Dans `src/evaluation/medical_metrics.py` :
- `is_valid_slice` rejette les coupes **vides** (ratio de pixels non nuls faible) ou **constantes** (écart-type bas).
- SSIM forcé à retourner **0.0** si images invalides.
- PSNR retourne 100 si images identiques (MSE=0, PSNR→inf), sinon 0.0 si invalide.

### 7.3. Post-filtrage côté interface
Dans `filter_valid_results(...)` :
- calcule métriques pour chaque résultat récupéré.
- exclut :
  - résultats invalides
  - résultats avec SSIM < `SSIM_DISPLAY_MIN`.

> Cet aspect est particulièrement intéressant pour votre mémoire : le système explicite une étape d’**assainissement des résultats** avant l’interprétation clinique.

---

## 8. Interfaces : restitution pour usage clinique et démonstration technique

### 8.1. Interface clinique
`src/interface/app_clinique.py` permet :
- upload d’une coupe IRM
- sélection de filtres cliniques (sexe, âge min/max, hôpital, diagnostic, modalité, etc.)
- choix du mode :
  - **CGR** (Guided) ou
  - **SupCon seul**
- affichage :
  - grade prédit + confiance (pour CGR)
  - résultats avec scores et images.

### 8.2. Interface technique (dashboard)
`src/interface/app_tech.py` fournit un cadre “preuve de recherche” :
1. **Pipeline Explorer**
   - temps encodage / recherche / métriques
   - affichage top-5 + vecteur latent
2. **Comparaison modèles**
   - baseline vs supcon vs guided (CGR)
   - radar chart : Cosinus, SSIM, HIST, Normalized PSNR
3. **Évaluation P@K**
   - lancement de l’évaluation stratifiée
   - figure et tableau détaillé
4. **Architecture**
   - description de chaque modèle et infrastructure.

---

## 9. Perspectives de rédaction (structure recommandée)

Vous pouvez structurer le mémoire en sections cohérentes avec le code :

### Chapitre A — Problématique et état de l’art
- CBIR en imagerie médicale
- limites des méthodes purely-visuelles
- intégration du contexte clinique
- problématique de déséquilibre BraTS.

### Chapitre B — Méthodologie
- prétraitement et extraction d’embeddings
- indexation Qdrant et rôle des métadonnées MongoDB
- architecture baseline et SupCon
- stratégie CGR (prédiction grade + filtrage).

### Chapitre C — Expérimentation
- protocole d’évaluation Precision@K stratifiée
- métriques image-based (SSIM/PSNR) et filtrage
- comparaison baseline vs supcon vs guided.

### Chapitre D — Résultats et discussion
- analyse Balanced Precision@K par grade
- interprétation de la contribution CGR
- discussion des garde-fous d’évaluation.

---

## 10. Indications pour compléter la rédaction avec des résultats

Ce projet génère (dans `evaluation_results/`) :
- `precision_at_k.json`
- `precision_at_k.png`

Pour un mémoire :
- insérez la figure Precision@K stratifiée
- discutez les valeurs macro-average Balanced Precision@K
- reliez aux enjeux de déséquilibre (hasard ≈ 1/3).

---

## 11. Liste des fichiers “cœur” à citer

- Moteurs :
  - `src/recherche/unified_engine.py`
  - `src/recherche/multimodal_search.py`
- Modèles :
  - `src/models/autoencoder.py`
  - `src/models/autoencoder_supervised.py`
  - `src/models/brats_classifier.py`
- Entraînement CGR :
  - `src/training/train_classifier.py`
  - `src/training/grade_dataset.py`
- Évaluation :
  - `src/evaluation/precision_at_k.py`
  - `src/evaluation/medical_metrics.py`
- Interfaces démonstratives :
  - `src/interface/app_tech.py`
  - `src/interface/app_clinique.py`
- Connexions BD :
  - `src/db/connections.py`

---

## 12. Remarques d’implémentation (utile pour la discussion)

- **Split par patient_id** lors de l’entraînement du classifieur (`BraTSGradeDataset`) : limite le data leakage.
- **CGR** : le filtrage dépend d’une prédiction de grade ; la performance réelle dépend donc de la qualité du classifieur.
- **Anti-images vides** : SSIM/PSNR sont robustifiées par des garde-fous (évite des métriques trompeuses sur images constantes).
- **Balanced Precision@K** : choix explicite pour ne pas être dominé par la classe majoritaire.

---

## 13. Conclusion (positionnement scientifique)

Le projet présente une chaîne end-to-end de CBMIR centrée sur une problématique de recherche :
- transformer l’image IRM en embeddings,
- effectuer une recherche vectorielle ANN,
- puis reconfigurer/filtrer la recherche selon un signal clinique (grade OMS) via CGR.

L’évaluation stratifiée et l’intégration de métriques image-based rendent le système adapté à un cadre de **validation scientifique** pour un mémoire Master.

