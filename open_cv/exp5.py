import cv2
import matplotlib.pyplot as plt
from skimage import exposure

image_path = '/content/moon_couple.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def display_images(images, titles):
    plt.figure(figsize=(6, 6))
    for i in range(len(images)):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

blurred = cv2.GaussianBlur(image, (5, 5), 0)

sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)

dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
rows,cols = image.shape
mask = np.zeros((rows,cols,2),np.uint8)
mask[rows//2-30:rows//2+30,cols//2-30:cols//2+30] = 1
low_pass = cv2.idft(np.fft.ifftshift(dft_shift*mask))
low_pass = cv2.magnitude(low_pass[:,:,0],low_pass[:,:,1])

display_images([image, blurred, sharpened, low_pass], ['Original', 'Gaussian Blurred',
                'Sharpened', 'Low Pass Filtered'])
