# download_medicalnet.py
import gdown, os

os.makedirs("./pretrain", exist_ok=True)

print("Téléchargement MedicalNet ResNet-50...")
gdown.download(
    "https://drive.google.com/uc?id=1399AbggS_Lg1W_YqNRotFTWAoribd9wv",
    "./pretrain/resnet_50_23dataset.pth",
    quiet=False
)
print("Téléchargé : ./pretrain/resnet_50_23dataset.pth")