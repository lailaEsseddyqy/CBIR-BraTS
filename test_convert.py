import torch
import cv2
import numpy as np
import os

def pt_to_png(pt_file_path, output_png_path):
    # 1. Charger le tenseur PyTorch
    tensor = torch.load(pt_file_path, weights_only=True)
    
    # 2. Le transformer en tableau Numpy (enlever les dimensions inutiles)
    img = tensor.squeeze().numpy()
    
    # 3. Normaliser les pixels entre 0 et 255 (format standard pour une image)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)
    
    # 4. Sauvegarder en PNG
    cv2.imwrite(output_png_path, img)
    print(f"✅ Image de test générée avec succès : {output_png_path}")

# --- TEST ---
# Remplace ces chemins par un vrai fichier de ton dossier brats_2d_slices
chemin_pt = "./Data/brats_subset_5k/slice_2334.pt" # Exemple
chemin_png = "test_image_brats2.png"

pt_to_png(chemin_pt, chemin_png)