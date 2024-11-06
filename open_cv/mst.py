# from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_image(image_path):
    img = cv2.imread(image_path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv_img)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)

    h = 255 - h
    s = np.clip(s * 1.2, 0, 255).astype(np.uint8)

    processed_hsv = cv2.merge((h, s, v))
    processed_bgr = cv2.cvtColor(processed_hsv, cv2.COLOR_HSV2BGR)

    return processed_bgr, processed_hsv

def display_images(original, processed_bgr, processed_hsv):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB))
    plt.title('HSV Image ')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(processed_hsv)
    plt.title('Enhanced Image (HSV)')
    plt.axis('off')

    plt.show()

image_path = '/content/CodeChefBadge.png'
original_image = cv2.imread(image_path)
processed_image_bgr, processed_image_hsv = process_image(image_path)

display_images(original_image, processed_image_bgr, processed_image_hsv)
