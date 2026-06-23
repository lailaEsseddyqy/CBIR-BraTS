import os
import torch
import matplotlib.pyplot as plt

# ── Configuration ────────────────────────────────────────────────
DATA_DIR = r"C:\Users\HP\Desktop\PROJET_PFE\CBIR-SYS\Data\brats_subset_5k"  
SAVE_PATH = "images/fig4_2_visual_comparison_top5.png"

def load_slice(slice_id):
    """
    Charge le tenseur PyTorch de la coupe IRM à partir de son slice_id.
    Retourne une matrice numpy 2D (128x128).
    """
    file_path = os.path.join(DATA_DIR, f"{slice_id}.pt")
    
    if os.path.exists(file_path):
        tensor = torch.load(file_path, weights_only=True)
        return tensor.squeeze().numpy()
    else:
        print(f"[Attention] Fichier introuvable : {file_path}")
        return torch.zeros((128, 128)).numpy()

def plot_visual_comparison(query_id, baseline_top5, radimagenet_top5, supcon_top5, combined_top5):
    """
    Génère et sauvegarde une figure matplotlib (5 lignes, 5 colonnes).
    """
    # Création de la figure globale (agrandie pour la 5ème ligne)
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))
    plt.subplots_adjust(hspace=0.3, wspace=0.1)

    # 1. Ligne 0 : Image Requête (placée au centre)
    for ax in axes[0]:
        ax.axis('off')
    
    ax_query = axes[0, 2]
    ax_query.imshow(load_slice(query_id), cmap='gray')
    ax_query.set_title(f"IMAGE REQUÊTE\n{query_id}", fontsize=12, fontweight='bold', color='darkred')
    ax_query.axis('on')
    ax_query.set_xticks([])
    ax_query.set_yticks([])

    # 2. Préparation des données pour les boucles
    models_data = [
        ("Baseline\n(256D)", baseline_top5, axes[1]),
        ("RadImageNet\n(2048D)", radimagenet_top5, axes[2]),
        ("SupCon\n(256D)", supcon_top5, axes[3]),
        ("Mode\nCombiné", combined_top5, axes[4])
    ]

    # 3. Remplissage des lignes de résultats
    for model_name, top5_ids, row_axes in models_data:
        # Ajout du nom du modèle à gauche de la ligne
        row_axes[0].text(-0.3, 0.5, model_name, fontsize=12, fontweight='bold', 
                         va='center', ha='right', transform=row_axes[0].transAxes)
        
        for i, slice_id in enumerate(top5_ids):
            ax = row_axes[i]
            img = load_slice(slice_id)
            ax.imshow(img, cmap='gray')
            
            short_id = "_".join(slice_id.split('_')[-2:]) if '_' in slice_id else slice_id
            ax.set_title(f"Rang {i+1}\n{short_id}", fontsize=10)
            
            ax.set_xticks([])
            ax.set_yticks([])

    # Titre global de la figure
    plt.suptitle("Comparaison Qualitative des Moteurs de Recherche (Top-5)", 
                 fontsize=16, fontweight='bold', y=0.92)
    
    # 4. Sauvegarde
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    plt.savefig(SAVE_PATH, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\n✅ Figure générée avec succès et sauvegardée dans : {SAVE_PATH}")


# ── Exécution de test ────────────────────────────────────────────
if __name__ == "__main__":
    requete_test = "slice_84" 
    
    res_baseline = ["slice_257", "slice_678", "slice_901", "slice_936", "slice_1049"]
    res_radimagenet = ["slice_1236", "slice_1308", "slice_1349", "slice_1434", "slice_1574"]
    res_supcon = ["slice_300", "slice_305", "slice_402", "slice_809", "slice_120"] # IDs fictifs pour le test
    res_combine = ["slice_257", "slice_1236", "slice_901", "slice_1308", "slice_300"]
    
    print("Génération de l'image comparative en cours...")
    plot_visual_comparison(requete_test, res_baseline, res_radimagenet, res_supcon, res_combine)