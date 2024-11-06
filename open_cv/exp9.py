import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/content/moon_couple.png',0)

_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image_with_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 3)

plt.imshow(image_with_contours)
plt.title('Contours (Boundary Linking)')
plt.show()
