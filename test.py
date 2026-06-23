import nibabel as nib
import numpy as np
from PIL import Image

# Charger le fichier .nii
img = nib.load("./Data/00000209_brain_flair.nii")
data = img.get_fdata() # volume 3D (H, W, Z)

# Prendre la coupe du milieu
z_mid = data.shape[2] // 2
slice_2d = data[:, :, z_mid]

# Normaliser et sauvegarder en PNG
slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min()) * 255
Image.fromarray(slice_2d.astype(np.uint8)).save("./Data/test_slice.png")
print(f"Image sauvegardée : ./Data/test_slice.png")