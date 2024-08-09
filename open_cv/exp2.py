import cv2
import numpy as np
import matplotlib.pyplot as plt
def convert_to_grayscale(image):
  R = image[:, :, 0]
  G = image[:,:, 1]
  B = image[:,:, 2]
  grayscale_image = 0.299 * R + 0.587 * G + 0.114 *  B
  return grayscale_image.astype (np.uint8)

image = cv2.imread('/content/CAR.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_gray_custom = convert_to_grayscale(image_rgb)

image_gray_cv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)


# line break

plt.figure(figsize=(12, 10))

plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title('Original RGB Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(image_gray_custom, cmap='gray')
plt.title('Grayscale Image (Formula) ')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(image_gray_cv, cmap='gray')
plt.title('Grayscale Image ')
plt.axis('off')

plt.tight_layout()
plt.show()
