import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
# Load the NIfTI file
file_pate = 'Data/00000209_brain_flair.nii'
nifti_img = nb.load(file_pate)

# Extract the matimatical Tensor
data_matrix = nifti_img.get_fdata()
# Data Prifiling
print("\n--- MRI DATA ANALYSIS ---")
print("Shape of the data matrix:", data_matrix.shape)

# Visualize 2D slices of the MRI data
slice_number = data_matrix.shape[2] // 2  
slice_2d = data_matrix[:, :, slice_number]

# Display using Matplotlib
plt.figure(figsize=(6, 6))
plt.imshow(slice_2d, cmap='gray') # Render the image in grayscale
plt.title(f"Axial Slice Number {slice_number}")
plt.axis('off') # Hide the coordinate axes
plt.show()
