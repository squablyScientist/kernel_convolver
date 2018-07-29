from skimage import io, color
from skimage.viewer import ImageViewer
import numpy as np
import sys


#TODO: allow this to generalize to rgb images instead of only grayscale ones

def convolve(kernel, img):
    output = np.empty_like(img)

    img = np.pad(img, ((0, 0), (0, 0)), mode='edge')

    for i in range(img.shape[0] - 2):
        for j in range(img.shape[1] - 2):
            
            #A lot is happening here:
            # performs element-wise multilplication w/ a submatrix of the original image and the kernel
            # takes the sum of the resulting matrix
            output[i, j] = (kernel * img[i:i+3, j:j+3]).sum()
    return output


sobely = [[-1, 0, 1],
          [-2, 0, 2,],
          [-1, 0, 1]]

sobelx = [[-1, -2, -1],
          [0, 0, 0,],
          [1, 2, 1]]

img = io.imread(sys.argv[1])
img = color.rgb2gray(img)

convoluted = convolve(sobelx, img)
ImageViewer(convoluted).show()


