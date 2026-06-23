
import nibabel as nib, numpy as np
from pathlib import Path

for p in sorted(Path("./Data/BraTS2021_Training_Data").glob("BraTS2021_*"))[:5]:
    seg = list(p.glob("*seg*"))
    if seg:
        arr = nib.load(str(seg[0])).get_fdata().astype(int)
        print(f"{p.name} → labels : {np.unique(arr)}")