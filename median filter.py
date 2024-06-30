
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def add_noise(img, noise_level):
    # Generate random noise of the same size as the image
    noise = np.random.normal(0, noise_level, img.shape)
    # Add noise to the image
    noisy_img = img + noise
    # Ensure pixel values are within the valid range (0 to 255 for uint8 images)
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

def median_filter(img, window_size):
    height, width = img.shape
    offset = window_size // 2
    filtered_img = np.zeros((height, width), dtype=np.uint8)

    for y in range(offset, height + offset):
        for x in range(offset, width + offset):
            window = img[y - offset:y + offset + 1, x - offset:x + offset + 1]
            sorted_window = np.sort(window.flatten())
            median_value = sorted_window[len(sorted_window) // 2]
            filtered_img[y - offset, x - offset] = median_value

    return filtered_img

input_image = cv2.imread("eclipse.jpg", cv2.IMREAD_GRAYSCALE)
window_size = 3  # Adjust the window size as needed (e.g., 3, 5, 7, etc.)
noise_level = 20  # Adjust the noise level as needed

# Add noise to the input image
noisy_image = add_noise(input_image, noise_level)

filtered_image = median_filter(noisy_image, window_size)

# Calculate PSNR
psnr_value = peak_signal_noise_ratio(input_image, filtered_image)

# Calculate SSIM
ssim_value, _ = structural_similarity(input_image, filtered_image, full=True)

cv2.imshow("Original Image", input_image)
cv2.imshow("Noisy Image", noisy_image)
cv2.imshow("Filtered Image", filtered_image)
print(f"PSNR: {psnr_value} dB")
print(f"SSIM: {ssim_value}")
cv2.waitKey(0)
cv2.destroyAllWindows()