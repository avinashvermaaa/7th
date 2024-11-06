import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/content/moon_couple.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
noise = np.random.normal(0, 25, gray_img.shape).astype(np.uint8)
noisy_img = cv2.add(gray_img, noise)
size = 15
motion_blur_kernel = np.zeros((size, size))
motion_blur_kernel[int((size - 1) / 2), :] = np.ones(size)
motion_blur_kernel = motion_blur_kernel / size
blurred_img = cv2.filter2D(noisy_img, -1, motion_blur_kernel)

def inverse_filter(img, kernel, epsilon=1e-8):
    dft_img = np.fft.fft2(img)
    dft_img_shifted = np.fft.fftshift(dft_img)
    dft_kernel = np.fft.fft2(kernel, s=img.shape)
    dft_kernel_shifted = np.fft.fftshift(dft_kernel)
    restored_img_shifted = dft_img_shifted / (dft_kernel_shifted + epsilon)
    restored_img = np.fft.ifft2(restored_img_shifted)
    restored_img_real = np.abs(restored_img)
    restored_img_real = np.clip(restored_img_real, 0, 255)
    return restored_img_real.astype(np.uint8)
restored_img = inverse_filter(blurred_img, motion_blur_kernel)

def display_images(images, titles):
    plt.figure(figsize=(10, 6))
    for i in range(len(images)):
        plt.subplot(1, 4, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
display_images([gray_img, noisy_img, blurred_img, ],
               ['Original Image', 'Noisy Image', 'Restored Image',])
