import nibabel as nb
import numpy as np

def find_bounding_box(volume_data):
    no_zero_coords = np.argwhere(volume_data > 0)   

    if no_zero_coords.size == 0 :
        return None, None
    min_coords = no_zero_coords.min(axis=0)
    max_coords = no_zero_coords.max(axis=0) + 1
    return min_coords, max_coords   



def crop_volume(volume_data, min_coords, max_coords):
    cropped_data = volume_data[
        min_coords[0]:max_coords[0], 
        min_coords[1]:max_coords[1], 
        min_coords[2]:max_coords[2]]
    return cropped_data

# Test the functions

file_path = 'Data/00000209_brain_flair.nii'
nifti_img = nb.load(file_path)
data_matrix = nifti_img.get_fdata()

print("\n--- Avant Cropping ---")
print("Dimensions:", data_matrix.shape)

# appliquer les fcts
min_c, max_c = find_bounding_box(data_matrix)
if min_c is not None and max_c is not None:
    cropped_data = crop_volume(data_matrix, min_c, max_c)
    print("\n--- Après Cropping ---")
    print("Dimensions:", cropped_data.shape)

    # Calculer  pourcentage  reduction
    volume_origin = data_matrix.size
    volume_cropped = cropped_data.size
    pourcentage_reduction = ((volume_origin - volume_cropped) / volume_origin) * 100
    print(f"Pourcentage de réduction: {pourcentage_reduction:.2f}%")
else:  
    print("Aucune RI trouvee pour le cropping.")