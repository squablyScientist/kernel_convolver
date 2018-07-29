import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io, color
import numpy as np
import sys


def convolve(kernel, img):
    output = np.empty_like(img)

    img = np.pad(img, ((0, 0), (0, 0)), mode='edge')

    for i in range(img.shape[0] - 2):
        for j in range(img.shape[1] - 2):
            
            #A lot is happening here:
            # performs element-wise multilplication w/ a submatrix of the original image and the kernel
            # takes the sum of the resulting matrix
            # performs ReLU on it in the form or min(max(x,0),255) to constrain the range to [0, 255]
            output[i, j] = min(max((kernel * img[i:i+3, j:j+3]).sum(), 0), 255)
    return output


kernel = [[0, -1, 0],
          [-1, 5, -1],
          [0, -1, 0]]


img = io.imread(sys.argv[1])
fig = plt.figure()
fig.add_subplot(1, 2, 1)
plt.imshow(img)

# Turn the image grayscale
img = color.rgb2gray(img)

fig.add_subplot(1, 2, 2)
convolved_img = convolve(kernel, img)
plt.imshow(convolved_img)

mpimg.imsave(sys.argv[2], convolved_img, cmap='gray')
plt.show()
