# DWT_compression_rgb_testing

Report on Image Compression and Denoising Workflow
Overview
This report outlines a workflow for compressing and denoising images using the Discrete Wavelet Transform (DWT) and Total Variation (TV) denoising. The process aims to reduce image size while maintaining quality, providing a measure of quality improvement through Signal-to-Noise Ratio (SNR).

Workflow Steps
Import Libraries: Essential libraries for image processing and visualization are imported, including NumPy, Matplotlib, PyWavelets, Scikit-Image, and PIL.

Load Image:

The image is loaded using PIL and converted into a NumPy array for processing.
Extract RGB DWT Coefficients:

Function: extract_rgb_coeff
Process: The image is separated into RGB channels, and DWT is applied to each channel independently.
Compress the Image:

Function: compress_image_dwt
Process: Each channel's DWT coefficients are compressed using thresholding based on a specified compression factor.
Reconstruct Image from DWT Coefficients:

Function: img_from_dwt_coeff
Process: The compressed coefficients of each RGB channel are combined to reconstruct the full RGB image.
Denoise the Recovered Image:

Function: denoise_image
Process: Total Variation denoising is applied to the reconstructed image to enhance its quality.
Calculate SNR:

Function: calculate_snr
Process: SNR is computed for the reconstructed image compared to the original image, as well as for the denoised image.
Calculate Sizes:

The sizes of the original image, compressed image, and denoised image are calculated to assess storage efficiency.
Visualize Results:

A series of plots are generated to display the original image, compressed image, denoised image, and the DWT coefficients for each channel, along with size and SNR information.
Total Loss of SNR:

The loss of SNR due to denoising is calculated and printed for evaluation.
Conclusion
The workflow effectively compresses and denoises images, achieving a balance between reduced file size and enhanced image quality. The use of DWT allows for efficient compression, while TV denoising further improves the visual quality of the reconstructed image.
