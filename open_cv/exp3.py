import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = '/content/moon_couple.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def display_images(images, titles):
    plt.figure(figsize=(7, 7))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

contrast_img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

log_img = (255 / np.log(1 + np.max(image))) * (np.log(image + 1)).astype(np.uint8)

gamma = 2.0
gamma_img = np.array(255 * (image / 255) ** gamma, dtype=np.uint8)

images = [image, contrast_img, log_img, gamma_img]
titles = ['Original Image', 'Contrast Stretched', 'Logarithmic Transformation',
          f'Gamma Correction (γ={gamma})']
display_images(images, titles)
