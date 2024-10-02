import numpy as np
import matplotlib.pyplot as plt
import pywt
from skimage.restoration import denoise_tv_chambolle
from PIL import Image

# Step 1: Compress the image using DWT with thresholding
def compress_image_dwt(image, wavelet='haar', compression_factor=0.1):
    # Perform DWT
    coeffs = pywt.dwt2(image, wavelet)

    # Flatten and apply thresholding to the approximation and details
    cA, (cH, cV, cD) = coeffs
    cA = np.where(np.abs(cA) < np.max(np.abs(cA)) * compression_factor, 0, cA)
    cH = np.where(np.abs(cH) < np.max(np.abs(cH)) * compression_factor, 0, cH)
    cV = np.where(np.abs(cV) < np.max(np.abs(cV)) * compression_factor, 0, cV)
    cD = np.where(np.abs(cD) < np.max(np.abs(cD)) * compression_factor, 0, cD)

    return (cA, (cH, cV, cD))

# Step 2: Recover the image using Inverse DWT
def recover_image_dwt(coeffs, wavelet='haar'):
    return pywt.idwt2(coeffs, wavelet)

# Step 3: Denoise the recovered image using Total Variation denoising
def denoise_image(image, weight=0.1):
    return denoise_tv_chambolle(image, weight=weight)

# Function to extract RGB DWT coefficients
def extract_rgb_coeff(img):
    img = img.convert('RGB')  # Ensure image is in RGB format
    mat_r = np.array(img)[:, :, 0]  # Red channel
    mat_g = np.array(img)[:, :, 1]  # Green channel
    mat_b = np.array(img)[:, :, 2]  # Blue channel
    
    coeffs_r = pywt.dwt2(mat_r, 'haar')
    coeffs_g = pywt.dwt2(mat_g, 'haar')
    coeffs_b = pywt.dwt2(mat_b, 'haar')

    return (coeffs_r, coeffs_g, coeffs_b)

# Function to recreate the image from RGB DWT coefficients
def img_from_dwt_coeff(coeff_dwt):
    (coeffs_r, coeffs_g, coeffs_b) = coeff_dwt
    reconstructed_r = pywt.idwt2(coeffs_r, 'haar')
    reconstructed_g = pywt.idwt2(coeffs_g, 'haar')
    reconstructed_b = pywt.idwt2(coeffs_b, 'haar')

    # Stack the channels back together
    reconstructed_image = np.stack((reconstructed_r, reconstructed_g, reconstructed_b), axis=-1)
    return Image.fromarray(np.uint8(reconstructed_image))

# Function to calculate SNR
def calculate_snr(original, noisy):
    mse = np.mean((original - noisy) ** 2)
    if mse == 0:
        return float('inf')  # If there is no noise, return infinity
    max_pixel_value = 255.0
    return 20 * np.log10(max_pixel_value / np.sqrt(mse))

# Load the image (replace 'image.jpg' with the path to your image)
image_path = 'image.jpg'  # Use PIL to load the image
image = Image.open(image_path)

# Convert to numpy array for size calculations
original_image_np = np.array(image)
original_size_kb = original_image_np.nbytes / 1024

# Extract RGB DWT coefficients
coeffs_rgb = extract_rgb_coeff(image)

# Compress the image by compressing each channel separately
compressed_coeffs = (
    compress_image_dwt(coeffs_rgb[0][0]),  # Red channel coefficients
    compress_image_dwt(coeffs_rgb[1][0]),  # Green channel coefficients
    compress_image_dwt(coeffs_rgb[2][0])   # Blue channel coefficients
)

# Calculate file sizes for each channel's compressed coefficients
compressed_size_kb = (
    np.array(compressed_coeffs[0][0]).nbytes / 1024 +  # Size of red channel approximation
    np.array(compressed_coeffs[0][1]).nbytes / 1024 +  # Size of red channel details
    np.array(compressed_coeffs[1][0]).nbytes / 1024 +  # Size of green channel approximation
    np.array(compressed_coeffs[1][1]).nbytes / 1024 +  # Size of green channel details
    np.array(compressed_coeffs[2][0]).nbytes / 1024 +  # Size of blue channel approximation
    np.array(compressed_coeffs[2][1]).nbytes / 1024     # Size of blue channel details
)

# Reconstruct the image from the DWT coefficients
reconstructed_image = img_from_dwt_coeff(compressed_coeffs)

# Convert the image back to numpy array (for denoising, etc.)
reconstructed_image_np = np.array(reconstructed_image)

# Denoise the recovered image to improve SNR (optional)
reconstructed_image_denoised = denoise_image(reconstructed_image_np, weight=0.1)

# Calculate sizes for reconstructed image
reconstructed_size_kb = reconstructed_image_np.nbytes / 1024

# Resize the original image to match the reconstructed image for SNR calculation
if original_image_np.shape != reconstructed_image_np.shape:
    original_image_resized = np.resize(original_image_np, reconstructed_image_np.shape)
else:
    original_image_resized = original_image_np

# Calculate SNRs
snr_reconstructed = calculate_snr(original_image_resized, reconstructed_image_np)
snr_denoised = calculate_snr(original_image_resized, reconstructed_image_denoised)

# Plot the results with SNR and size displayed
plt.figure(figsize=(24, 18))

# Original Image
plt.subplot(4, 3, 1)
plt.imshow(image)
plt.title(f'Original Image\nSize: {original_size_kb:.2f} KB\nSNR: Inf (Perfect)')
plt.axis('off')

# Compressed Image
plt.subplot(4, 3, 2)
plt.imshow(reconstructed_image)
plt.title(f'Compressed Image\nSize: {compressed_size_kb:.2f} KB\nSNR: {snr_reconstructed:.2f} dB')
plt.axis('off')

# Reconstructed Image from DWT
plt.subplot(4, 3, 3)
plt.imshow(reconstructed_image_denoised)
plt.title(f'Reconstructed Image (RGB DWT)\nSize: {reconstructed_size_kb:.2f} KB\nSNR: {snr_denoised:.2f} dB')
plt.axis('off')

# DWT Coefficients for Red Channel
plt.subplot(4, 3, 4)
plt.imshow(coeffs_rgb[0][0], cmap='gray')
plt.title('DWT Coefficients (Red Channel)\nApproximation')
plt.axis('off')

plt.subplot(4, 3, 5)
plt.imshow(coeffs_rgb[0][1][0], cmap='gray')
plt.title('DWT Coefficients (Red Channel)\nHorizontal Detail')
plt.axis('off')

plt.subplot(4, 3, 6)
plt.imshow(coeffs_rgb[0][1][1], cmap='gray')
plt.title('DWT Coefficients (Red Channel)\nVertical Detail')
plt.axis('off')

# DWT Coefficients for Green Channel
plt.subplot(4, 3, 7)
plt.imshow(coeffs_rgb[1][0], cmap='gray')
plt.title('DWT Coefficients (Green Channel)\nApproximation')
plt.axis('off')

plt.subplot(4, 3, 8)
plt.imshow(coeffs_rgb[1][1][0], cmap='gray')
plt.title('DWT Coefficients (Green Channel)\nHorizontal Detail')
plt.axis('off')

plt.subplot(4, 3, 9)
plt.imshow(coeffs_rgb[1][1][1], cmap='gray')
plt.title('DWT Coefficients (Green Channel)\nVertical Detail')
plt.axis('off')

# DWT Coefficients for Blue Channel
plt.subplot(4, 3, 10)
plt.imshow(coeffs_rgb[2][0], cmap='gray')
plt.title('DWT Coefficients (Blue Channel)\nApproximation')
plt.axis('off')

plt.subplot(4, 3, 11)
plt.imshow(coeffs_rgb[2][1][0], cmap='gray')
plt.title('DWT Coefficients (Blue Channel)\nHorizontal Detail')
plt.axis('off')

plt.subplot(4, 3, 12)
plt.imshow(coeffs_rgb[2][1][1], cmap='gray')
plt.title('DWT Coefficients (Blue Channel)\nVertical Detail')
plt.axis('off')

plt.tight_layout()
plt.show()

# Print total loss of SNR
total_loss_snr = snr_reconstructed - snr_denoised
print(f'Total Loss of SNR: {total_loss_snr:.2f} dB')
