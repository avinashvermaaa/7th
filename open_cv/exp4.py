import cv2
import matplotlib.pyplot as plt
from skimage import exposure  # Make sure to import exposure from skimage

# Load the image
image_path = '/content/moon_couple.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
reference = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Function to display images in a 1x3 matrix
def display_images(images, titles):
    plt.figure(figsize=(10, 5))  # Adjust the figure size for horizontal display
    for i in range(len(images)):
        plt.subplot(1, 3, i + 1)  # Create a 1x3 subplot
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Match histograms
matched_img = exposure.match_histograms(image, reference)

# Histogram equalization
specified_img = cv2.equalizeHist(image)

# Display all images in a 1x3 matrix
display_images([image, matched_img, specified_img], ['Original', 'Matched',
                'Histogram Specified'])
