import matplotlib.image as mpimg
from skimage import io, color
from skimage.viewer import ImageViewer
import cv2
import numpy as np
import sys


#TODO: allow this to generalize to rgb images instead of only grayscale ones

def convolve(kernel, img):
    output = np.empty_like(img)

    # Pads the image by extending it by 1 pixel on all sides
    img = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode='edge')

    for i in range(img.shape[0] - 2):
        for j in range(img.shape[1] - 2):
            for k in range(3): 
                #A lot is happening here:
                # performs element-wise multilplication w/ a submatrix of the original image and the kernel
                # takes the sum of the resulting matrix
                output[i, j, k] =min(max((kernel[k] * img[i:i+3, j:j+3, k]).sum(), 0), 255)
    return output


sobely =  np.array([[-1, 0, 1],
          [-2, 0, 2,],
          [-1, 0, 1]])

sobelx =  np.array([[-1, -2, -1],
          [0, 0, 0,],
          [1, 2, 1]])

blur = np.multiply(1/9, np.full((3, 3, 3), 1))
        
img = cv2.imread(sys.argv[1], 1)


laplacian = np.full((3, 3), -1)
laplacian[1, 1] = 8


laplacian_color = [laplacian, laplacian, laplacian]

convoluted = convolve(blur, img)
ImageViewer(convoluted).show()
io.imsave('test.png', convoluted)

