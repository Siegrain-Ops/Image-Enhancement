import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np
from model import EnhancementCNN  
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

#Paths
MODEL_PATH = "../models/enhancement_cnn.pth"
INPUT_IMAGE_PATH = "../data/test_images/test.jpg"
OUTPUT_IMAGE_PATH = "../data/test_images/enhanced_test.jpg"

#device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model load
model = EnhancementCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()  #set model in evaluation mode

#test image tranformation
transform = transforms.Compose([
    transforms.Resize((512, 512)), 
    transforms.ToTensor()
])

#test image load
raw_image = Image.open(INPUT_IMAGE_PATH).convert("RGB")
raw_tensor = transform(raw_image).unsqueeze(0).to(device)  #tranform image to tensor

#upgrading image
with torch.no_grad():
    enhanced_tensor = model(raw_tensor)
    enhanced_tensor = torch.clamp(enhanced_tensor, 0, 1)  #make sure to get all tensor values in the span 0-1

#Convert tensor to image PIL
enhanced_image = transforms.ToPILImage()(enhanced_tensor.squeeze(0).cpu())

#Upscaling with Real-ESRGAN
model_esrgan = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
esrgan = RealESRGANer(scale=4, model_path='../models/weights_esrgan/RealESRGAN_x4plus.pth', model=model_esrgan, tile=512, tile_pad=10, pre_pad=0, half=True)

# Convert PIL to OpenCV for ESRGAN
cv2_image = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)
upscaled_image, _ = esrgan.enhance(cv2_image, outscale=4)

#saving image
cv2.imwrite(OUTPUT_IMAGE_PATH, upscaled_image)

print(f"Image upgraded and upscaled saved in: {OUTPUT_IMAGE_PATH}")
