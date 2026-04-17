import numpy as np
import nibabel as nb
import skfuzzy as fuzz
import matplotlib.pyplot as plt 

def find_bounding_box(volume_data):
    no_zero_coords = np.argwhere(volume_data > 0)   

    if no_zero_coords.size == 0 :
        return None, None
    min_coords = no_zero_coords.min(axis=0)
    max_coords = no_zero_coords.max(axis=0) + 1
    return min_coords, max_coords

file_path = 'Data/00000209_brain_flair.nii'
nifti_img = nb.load(file_path)
data_matrix = nifti_img.get_fdata()

min_c, max_c = find_bounding_box(data_matrix)
cropped_data = data_matrix[
    min_c[0]:max_c[0],
    min_c[1]:max_c[1],
    min_c[2]:max_c[2] ]

z_mid = cropped_data.shape[2] // 2
slice_2d = cropped_data[:, :, z_mid]

shape_2d = slice_2d.shape
data_flat = slice_2d.reshape(1, -1)

print(f"Lancement du FCM sur la coupe {z_mid}")
n_clusters = 4
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data_flat, c=n_clusters, m=2, error=0.005, maxiter=100, init=None
)

segmented_flat = np.argmax(u, axis=0)
segmented_2d = segmented_flat.reshape(shape_2d)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(slice_2d, cmap='gray')
axes[0].set_title('Coupe recadrée (Originale)')
axes[0].axis('off')

# L'utilisation de 'nipy_spectral' permet de donner des couleurs très vives aux clusters
axes[1].imshow(segmented_2d, cmap='nipy_spectral')
axes[1].set_title(f'Segmentation FCM ({n_clusters} classes)')
axes[1].axis('off')

print("Calcul termine  Affichage de l'image en cours...")
plt.show()