clc
clear all
close all

% Read the input image
input_image = imread('ANPR garage 1.jpg');
input_image = rgb2gray(input_image); % Convert to grayscale if needed

% Parameters
window_size = 3; % Adjust the window size as needed (e.g., 3, 5, 7, etc.)
noise_density = 0.2; % Adjust the noise density as needed

% Add salt and pepper noise to the input image
noisy_image = imnoise(input_image, 'salt & pepper', noise_density);

% Apply median filter
filtered_image = medfilt2(noisy_image, [window_size window_size]);

% Calculate PSNR
psnr_value = psnr(filtered_image, input_image);

% Calculate SSIM
ssim_value = ssim(filtered_image, input_image);

% Display images and results
subplot(1,3,1), imshow(input_image), title('Original Image');
subplot(1,3,2), imshow(noisy_image), title('Noisy Image');
subplot(1,3,3), imshow(filtered_image), title('Filtered Image');

fprintf('PSNR: %.2f dB\n', psnr_value);
fprintf('SSIM: %.4f\n', ssim_value);
