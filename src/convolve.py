import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io, color
import numpy as np
import sys


def convolve(kernel, img):
    output = np.empty_like(img)
    img_padding = np.zeros((img.shape[0] + 2, img.shape[1] + 2))

    # Places original image on top of the padding
    img_padding[1:-1, 1:-1] = img

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i, j] = (kernel * img_padding[i:i+3, j:j+3]).sum()
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
