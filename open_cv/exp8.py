import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/content/moon_couple.png')

edge = cv2.Canny(img, 100, 200)

sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

laplacian = cv2.Laplacian(img, cv2.CV_64F)

def display_images(images, titles):
    plt.figure(figsize=(10, 6))
    for i in range(len(images)):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

display_images([image,edge,sobel_x,laplacian,],
               ['Original Image', 'Canny Edges','Sobel X','Laplacian',])
