import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, Subset
from dataset import ImageEnhancementDataset, transform
from model import EnhancementCNN
import matplotlib.pyplot as plt
import torchmetrics.image
import os


# Configuration
EPOCHS = 50
BATCH_SIZE = 64  
LEARNING_RATE = 1e-4
SAVE_PATH = "../data/models/enhancement_cnn.pth"  

# Dataset Path
DATASET_PATH = "../data"
RAW_DIR = os.path.join(DATASET_PATH, "raw")
PROCESSED_DIR = os.path.join(DATASET_PATH, "processed")

# Check if dataset path exist (if you are running on google drive with colab or whatsoever)
if not os.path.exists(RAW_DIR) or not os.path.exists(PROCESSED_DIR):
    raise FileNotFoundError("Dataset not found, make sure to upload data.")
    
# Def Perceptual Loss with VGG16
class PerceptualLoss(nn.Module):
    def __init__(self, layer=8):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features[:layer].eval()  # Takes only the first 8 layers
        for param in vgg.parameters():
            param.requires_grad = False  # We don't need to train VGG16
        self.vgg = vgg
        self.criterion = nn.MSELoss() #Mean Squared Error Loss

    def forward(self, x, y):
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        return self.criterion(x_features, y_features)

#GPU Optimization
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

if __name__ == "__main__":
    #Dataset loading with subset to not fill Ram
    full_dataset = ImageEnhancementDataset(raw_dir=RAW_DIR, enhanced_dir=PROCESSED_DIR, transform=transform)
    dataset = Subset(full_dataset, range(min(5000, len(full_dataset))))  #Max 5000 images
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    #Initialize Model and GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancementCNN().to(device)
    
    #Defining of loss function
    mse_loss = nn.MSELoss()
    perceptual_loss = PerceptualLoss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) #Adam optimizer
    scaler = torch.amp.GradScaler()  #Mixed Precision Training

    #Defining of methods to calculate SSIM and PSNR
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure().to(device)
    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio().to(device)

    #Storing of metrics
    losses, ssim_scores, psnr_scores = [], [], []

    #Training loop
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS} - Start Training")
        model.train()
        epoch_loss, epoch_ssim, epoch_psnr = 0, 0, 0

        for batch_idx, (raw_images, enhanced_images) in enumerate(dataloader):
            raw_images, enhanced_images = raw_images.to(device), enhanced_images.to(device)
            optimizer.zero_grad()

            try:
                with torch.amp.autocast(device_type="cuda"):  #Mixed Precision for faster training
                    outputs = model(raw_images)
                    
                    #Combination of MSE loss and Perceptual Loss
                    loss_mse = mse_loss(outputs, enhanced_images)
                    loss_perceptual = perceptual_loss(outputs, enhanced_images)
                    loss = loss_mse + 0.1 * loss_perceptual  #Balancing
                    
                #Check if loss is NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"WARNING: Loss has values NaN or Inf inside the batch {batch_idx}. Training interrupted.")
                    break  #Interrupt

                scaler.scale(loss).backward() #Scaler do Loss scaling before backpropagation
                scaler.step(optimizer)
                scaler.update() #dinamically update scaler

                epoch_loss += loss.item()
                epoch_ssim += ssim_metric(outputs, enhanced_images).item()
                epoch_psnr += psnr_metric(outputs, enhanced_images).item()

            except RuntimeError as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue

        #Saving Metrics
        avg_loss = epoch_loss / len(dataloader)
        avg_ssim = epoch_ssim / len(dataloader)
        avg_psnr = epoch_psnr / len(dataloader)
        losses.append(avg_loss)
        ssim_scores.append(avg_ssim)
        psnr_scores.append(avg_psnr)

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.6f}, SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.4f}")

        #Check saving folder
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        
        #Saving Model
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"Model saved in: {SAVE_PATH}")

    #Plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(range(1, EPOCHS + 1), losses, label='Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss during Training")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(range(1, EPOCHS + 1), ssim_scores, label='SSIM', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("SSIM score")
    plt.title("SSIM during Training")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(range(1, EPOCHS + 1), psnr_scores, label='PSNR', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("PSNR score")
    plt.title("PSNR during Training")
    plt.legend()

    plt.show()
