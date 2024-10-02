import numpy as np
import cv2
import matplotlib.pyplot as plt
import pywt
from skimage.restoration import denoise_tv_chambolle
from PIL import Image
#import utils as util  # Assuming utils is in the same directory
 
# Step 1: Compress the image using DWT with thresholding
def compress_image_dwt(image, wavelet='haar', level=1, compression_factor=0.1):
    # Perform DWT
    coeffs = pywt.wavedec2(image, wavelet, level=level)
 
    # Flatten and apply thresholding to the approximation and details
    coeffs_thresholded = []
    for i, coeff in enumerate(coeffs):
        if isinstance(coeff, tuple):  # This is the detail coefficients
            cA, (cH, cV, cD) = coeff
            # Apply thresholding
            cA = np.where(np.abs(cA) < np.max(np.abs(cA)) * compression_factor, 0, cA)
            cH = np.where(np.abs(cH) < np.max(np.abs(cH)) * compression_factor, 0, cH)
            cV = np.where(np.abs(cV) < np.max(np.abs(cV)) * compression_factor, 0, cV)
            cD = np.where(np.abs(cD) < np.max(np.abs(cD)) * compression_factor, 0, cD)
            coeffs_thresholded.append((cA, (cH, cV, cD)))
        else:  # This is the approximation coefficients at the highest level
            coeffs_thresholded.append(np.where(np.abs(coeff) < np.max(np.abs(coeff)) * compression_factor, 0, coeff))
 
    return coeffs_thresholded
 
# Step 2: Recover the image using Inverse DWT
def recover_image_dwt(coeffs_thresholded, wavelet='haar'):
    # Perform inverse DWT
    return pywt.waverec2(coeffs_thresholded, wavelet)
 
# Step 3: Denoise the recovered image using Total Variation denoising
def denoise_image(image, weight=0.1):
    # Apply Total Variation denoising
    return denoise_tv_chambolle(image, weight=weight)
 
# Function to extract RGB DWT coefficients
def extract_rgb_coeff(img):
    """
    Returns RGB dwt applied coefficients tuple
    Parameters
    ----------
    img: PIL Image
    Returns
    -------
    (coeffs_r, coeffs_g, coeffs_b):
        RGB coefficients with Discrete Wavelet Transform Applied
    """
    (width, height) = img.size
    img = img.copy()
 
    mat_r = np.empty((width, height))
    mat_g = np.empty((width, height))
    mat_b = np.empty((width, height))
 
    for i in range(width):
        for j in range(height):
            (r, g, b) = img.getpixel((i, j))
            mat_r[i, j] = r
            mat_g[i, j] = g
            mat_b[i, j] = b
 
    coeffs_r = pywt.dwt2(mat_r, 'haar')
    coeffs_g = pywt.dwt2(mat_g, 'haar')
    coeffs_b = pywt.dwt2(mat_b, 'haar')
 
    return (coeffs_r, coeffs_g, coeffs_b)
 
# Function to recreate the image from RGB DWT coefficients
def img_from_dwt_coeff(coeff_dwt):
    """
    Returns Image recreated from dwt coefficients
    Parameters
    ----------
    (coeffs_r, coeffs_g, coeffs_b):
        RGB coefficients with Discrete Wavelet Transform Applied
    Returns
    -------
    Image from dwt coefficients
    """
    (coeffs_r, coeffs_g, coeffs_b) = coeff_dwt
    (width, height) = (len(coeffs_r[0]), len(coeffs_r[0][0]))
 
    # Channel Red
    cARed = np.array(coeffs_r[0])
    cHRed = np.array(coeffs_r[1][0])
    cVRed = np.array(coeffs_r[1][1])
    cDRed = np.array(coeffs_r[1][2])
 
    # Channel Green
    cAGreen = np.array(coeffs_g[0])
    cHGreen = np.array(coeffs_g[1][0])
    cVGreen = np.array(coeffs_g[1][1])
    cDGreen = np.array(coeffs_g[1][2])
 
    # Channel Blue
    cABlue = np.array(coeffs_b[0])
    cHBlue = np.array(coeffs_b[1][0])
    cVBlue = np.array(coeffs_b[1][1])
    cDBlue = np.array(coeffs_b[1][2])
 
    # Image object init
    dwt_img = Image.new('RGB', (width, height), (0, 0, 20))
 
    # Reconstruct each channel
    for i in range(width):
        for j in range(height):
            R = (cARed[i][j] / np.max(np.abs(cARed))) * 100.0
            G = (cAGreen[i][j] / np.max(np.abs(cAGreen))) * 100.0
            B = (cABlue[i][j] / np.max(np.abs(cABlue))) * 100.0
            new_value = (int(R), int(G), int(B))
            dwt_img.putpixel((i, j), new_value)
 
    return dwt_img
 
# Load the image (replace 'image.jpg' with the path to your image)
image = Image.open('image.jpg')  # Use PIL to load the image
 
# Extract RGB DWT coefficients
coeffs_rgb = extract_rgb_coeff(image)
 
# Recreate the image from RGB DWT coefficients
reconstructed_image = img_from_dwt_coeff(coeffs_rgb)
 
# Convert the image back to numpy array (for denoising, etc.)
reconstructed_image_np = np.array(reconstructed_image)
 
# Denoise the recovered image to improve SNR (optional)
reconstructed_image_denoised = denoise_image(reconstructed_image_np, weight=0.1)
 
# Plot the results
plt.figure(figsize=(15, 8))
 
# Original Image
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')
 
# Reconstructed Image from DWT
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image_denoised)
plt.title('Reconstructed Image (RGB DWT)')
plt.axis('off')
 
plt.tight_layout()
plt.show()