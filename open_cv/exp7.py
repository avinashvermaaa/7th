import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/content/moon_couple.png')
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)
cv2.imwrite('erosion.png', erosion)
dilation = cv2.dilate(img, kernel, iterations=1)
cv2.imwrite('dilation.png', dilation)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imwrite('opening.png', opening)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('closing.png', closing)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
cv2.imwrite('gradient.png', gradient)

def display_images(images, titles):
    plt.figure(figsize=(10, 6))
    for i in range(len(images)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

display_images([image,erosion,dilation,opening,closing,gradient,],
               ['Original Image', 'Erosion','Dilation','Opening','Closing','Gradient',])
