import cv2
import matplotlib.pyplot as plt
from project.lesson_functions import *

image = mpimg.imread('../test_images/test1.jpg')
image = image[350:500, 1000:, :]
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
luv = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
xyz = cv2.cvtColor(image, cv2.COLOR_RGB2XYZ)
yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
ycc = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

color_spaces_names = ['RGB', 'HSV', 'HLS', 'LUV', 'LAB', 'XYZ', 'YUV', 'YCrCb']
color_spaces = [image, hsv, hls, luv, lab, xyz, yuv, ycc]

rows = len(color_spaces)

fig, axis = plt.subplots(rows, 4, figsize=(16, 3*rows))
for row, colorspace in enumerate(color_spaces):
    axis[row, 0].set_title(color_spaces_names[row])
    axis[row, 0].imshow(colorspace)
    axis[row, 0].axis('off')
    for ch in range(3):
        n, bins, patches = plt.hist(colorspace[:,:,ch].flatten()/np.max(colorspace), bins=32,  range=(0, 1))
        axis[row, ch + 1].set_title('CH-%d'%(ch+1))
        axis[row, ch + 1].bar(bins[:-1]*255, n)

plt.savefig('../test_output/colorspace_histogram.jpg')