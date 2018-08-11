import matplotlib.pyplot as plt
from skimage import io, color
import numpy as np
import cv2
import sys


#TODO: allow this to generalize to rgb images instead of only grayscale ones


sobelx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

sobelx2 = np.flipud(sobelx)

sobely = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])

sobely2 = np.flipud(sobely)

blur = np.multiply((1 / 9),
                   np.array([[1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1]]))


def convolve(kernel, img):
    output = np.empty_like(img)

    # Pads the image by extending it by 1 pixel on all sides
    img = np.pad(img, ((1, 1), (1, 1)), mode='edge')

    for i in range(img.shape[0] - 2):
        for j in range(img.shape[1] - 2):
            
            # A lot is happening here:
            #   performs element-wise multiplication w/ a submatrix of the original image and the kernel
            #   takes the sum of the resulting matrix
            output[i, j] = min(max((kernel * img[i:i+3, j:j+3]).sum(), 0), 255)
    return output


img_in = cv2.imread(sys.argv[1], 0)

convoluted = convolve(sobely, img_in)
convoluted2 = convolve(sobely2, img_in)
convoluted3 = convolve(sobelx, img_in)
convoluted4 = convolve(sobelx2, img_in)

# Combine both sobel
convoluted_final = np.empty_like(img_in)
for i in range(img_in.shape[0]):
    for j in range(img_in.shape[1]):
        convoluted_final[i, j] = (convoluted[i, j] + convoluted2[i, j])/2 + (convoluted3[i, j] + convoluted4[i, j])/2

blurred = convolve(blur, img_in)

plt.imshow(blurred, cmap='gray')
plt.show()
io.imsave('test.png', convoluted)

