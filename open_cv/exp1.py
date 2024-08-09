from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# line 2

original_image = Image.open('/content/me.jpeg')
rgb_image = original_image.convert('RGB')
rgb_image.save('/content/me.jpeg')
gray_image = original_image.convert('L')
original_image_cv= np.array(original_image)
hsv_image= cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2HSV)


# line 3

plt.figure(figsize=(11,11))
plt.subplot(1,3,1)
plt.title('RGB Image')
plt.imshow(rgb_image)
plt.subplot(1,3,2)
plt.title('Grayscale Image')
plt.imshow(gray_image, cmap='gray')
plt.subplot(1,3,3)
plt.title('HSV Image')
plt.imshow(hsv_image)
plt.show()
