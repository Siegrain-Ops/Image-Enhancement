import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

#Paths
DATASET_PATH = "../data"
RAW_DIR = os.path.join(DATASET_PATH, "raw")
PROCESSED_DIR = os.path.join(DATASET_PATH, "processed")

#Check Dataset
if not os.path.exists(RAW_DIR) or not os.path.exists(PROCESSED_DIR):
    raise FileNotFoundError("Dataset not found, make sure to upload data.")

#main function
class ImageEnhancementDataset(Dataset):
    def __init__(self, raw_dir=RAW_DIR, enhanced_dir=PROCESSED_DIR, transform=None):
        #loading raw and processed images
        self.raw_dir = raw_dir
        self.enhanced_dir = enhanced_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(raw_dir) if f.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        raw_img_name = os.path.join(self.raw_dir, self.image_files[idx])
        enhanced_img_name = os.path.join(self.enhanced_dir, self.image_files[idx])

        raw_image = Image.open(raw_img_name).convert("RGB")
        enhanced_image = Image.open(enhanced_img_name).convert("RGB")

        print(f"Loaded RAW: {raw_img_name}, Dim: {raw_image.size}")  
        print(f"Loaded PROCESSED: {enhanced_img_name}, Dim: {enhanced_image.size}")

        #Apply tranformations
        if self.transform:
            raw_image = self.transform(raw_image)
            enhanced_image = self.transform(enhanced_image)

        return raw_image, enhanced_image

#Trasformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),  #Resize at 512x512
    transforms.ToTensor()
])

#Test dataset.py
if __name__ == "__main__":
    dataset = ImageEnhancementDataset(transform=transform)
    print(f"Dataset size: {len(dataset)}")
    sample_raw, sample_enhanced = dataset[0]
    print(f"Raw image shape: {sample_raw.shape}")
    print(f"Enhanced image shape: {sample_enhanced.shape}")
