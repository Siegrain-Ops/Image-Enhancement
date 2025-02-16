This project implements a Convolutional Neural Network (CNN) for image enhancement, designed to improve image quality by reducing noise and restoring details. 
The model is trained on datasets such as MIT-Adobe FiveK and GOPRO Blur Dataset, but it can be trained on any dataset containing raw and processed image pairs.






FEATURES

- End-to-End Image Enhancement: No need for manual preprocessing—just input a low-quality image and get an improved version.
- Trainable on Any Dataset: The model is flexible and can be trained on different datasets for tasks like denoising, deblurring, and super-resolution.
- Custom CNN Architecture: Unlike GANs or Transformer-based models, this CNN focuses on learning direct pixel transformations for quality enhancement.
- Perceptual Loss with VGG16: Uses MSE loss combined with perceptual loss (based on VGG16) to improve visual fidelity.




MODEL ARCHITECTURE

The model follows a deep convolutional structure with multiple layers for feature extraction, enhancement, and reconstruction.

- FIRST BLOCK: Convolutional Layers (Feature Extraction) 

The first few layers extract low-level features such as edges, textures, and gradients.
Each convolutional layer applies 3x3 or 5x5 filters followed by Batch Normalization and ReLU activation for non-linearity.

- SECOND BLOCK: Deep Feature Processing (Enhancement)

Midway through the network, the model learns how to restore lost details from blurred or low-quality images.
The deeper layers work on removing noise, correcting colors, and improving contrast.

- THIRD BLOCK: Upscaling & Reconstruction

The final layers reconstruct the enhanced image.
Instead of using upsampling layers like in GANs, we rely on convolutional refinement to enhance sharpness while maintaining original resolution.



LOSS FUNCTION USED

To improve both pixel accuracy and perceptual quality, we use a hybrid loss function:

- Mean Squared Error (MSE) Loss
Standard pixel-wise loss that minimizes the difference between the output and ground truth.
Ensures that the enhanced image remains structurally faithful to the original.

- Perceptual Loss (VGG16)
Uses a pre-trained VGG16 network to compare high-level feature representations instead of just pixel values.
Helps retain fine details and produces sharper, more natural images.

- Structural Similarity Index (SSIM) & PSNR (Evaluation Metrics)
SSIM measures how perceptually similar the enhanced image is to the target.
PSNR quantifies the level of detail recovery.
