import nibabel as nb
import numpy as np

# Load the NIfTI file
file_path = 'Data/BraTS2021_00495_seg.nii'

seg_img = nb.load(file_path)
seg_data = seg_img.get_fdata()

# Data Profiling
labels, counts = np.unique(seg_data, return_counts=True)
total_voxels = seg_data.size

# Print pourcentage 
print("\n -- Distribution des classes (DESEQUILIBRE) --")
print("Volume total de voxels:", total_voxels)

nams_classes = {
    0: "Fond noir + Cerveau sain",
    1: "Cœur nécrotique (Label 1)",
    2: "Œdème (Label 2)",
    3: "Tumeur active (Label 4)"
}

for label, count in zip(labels, counts):
    percentage = (count / total_voxels) * 100
    class_name = nams_classes.get(label, "Inconnu")
    print(f"Classe {label} ({class_name}): {count} voxels ({percentage:.2f}%)")