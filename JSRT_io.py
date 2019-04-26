#referred from https://stackoverflow.com/questions/32946436/read-img-medical-image-without-header-in-python on Feb 5, 2019
#referred from https://github.com/harishanand95/jsrt-parser/blob/master/jsrt.py on Feb 5, 2019
import matplotlib.pyplot as plt
import numpy as np
from csv import reader, excel_tab
from scipy import ndimage
import cv2
from PIL import Image
# Parameters.
input_filename = "JSRT/All247/JPCLN010.png"
mask_filename = "JSRT/scratch/All247imagesMask/right_lung/JPCLN010.png"
# input_filename = "JSRT/Normal1/JPCNN001.IMG"
shape_in = (2048, 2048) # matrix size
shape_mask = (1024,1024)
new_shape = (512, 512)
dtype = np.dtype('>u2') # big-endian unsigned integer (16bit)
output_filename = "JPCLN144.png"
output_filename_mask = "JPCLN144_mask.png"

# Reading.
# fid_in = open(input_filename, 'r')
# fid_mask

#fetch info
# content = reader(fid, dialect=excel_tab)
# print(content)
# for row in content:
#     print (row)

#plotting
# data = np.fromfile(fid_in, dtype)
# print('shape data', data.shape)
# image = data.reshape(shape_in)
# im = Image.fromarray(image)
# im.save(output_filename)

#print('x cor is', image)
# Display.
# plt.imshow(image, cmap = "gray")
# plt.savefig(output_filename)
# cv2.imread(output_filename)
# image = cv2.resize(image, new_shape)
# cv2.imwrite(output_filename, image)

# plt.show()

image = cv2.imread(input_filename)
image = cv2.resize(image, new_shape)
cv2.imwrite(output_filename, image)

image = cv2.imread(mask_filename)
image = cv2.resize(image, new_shape)
cv2.imwrite(output_filename_mask, image)


# image = im_array.reshape(shape_mask)
# plt.imshow(image, cmap = "gray")
# plt.savefig(output_filename_mask)
# plt.show()

# cv2.imread(output_filename_mask)


# image = im_array.reshape(shape_mask)
# plt.imshow(image, cmap = "gray")
# plt.savefig(output_filename_mask)
# plt.show()

# cv2.imread(output_filename_mask)
# image1 = ndimage.imread(output_filename_mask)
# image1 = cv2.imread(mask_filename)
# image1 = cv2.resize(image1, new_shape)
# cv2.imwrite(output_filename_mask, image1)

# im = Image.open(mask_filename)
# bg = Image.new("L", im.size)
# bg.paste(im, (0,0), im)
# bg.save(output_filename_mask, quality=95)
