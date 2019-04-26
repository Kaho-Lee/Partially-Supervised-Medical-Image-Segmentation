import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mnet_utils import BW_img, disc_crop, mk_dir, return_list
from Utils import *

test_data = 'UNetJSRT/val/IMG/'
test_mask = 'UNetJSRT/val/'
test_pred = 'UNetJSRT/test/'
root_path = 'UNetJSRT/'
test_files = return_list(test_data, '.png')
image = cv2.imread(test_data+test_files[19])

mask_image_right_lung = cv2.imread(test_mask+'right_lung/'+test_files[19])
mask_image_left_lung = cv2.imread(test_mask+'left_lung/'+test_files[19])
mask_image_heart = cv2.imread(test_mask+'heart/'+test_files[19])
mask_image_left_clavicle = cv2.imread(test_mask+'left_clavicle/'+test_files[19])
mask_image_right_clavicle = cv2.imread(test_mask+'right_clavicle/'+test_files[19])

predict_mask_left_lung = cv2.imread(test_pred+'class2/'+test_files[19])
predict_mask_right_lung = cv2.imread(test_pred+'class1/'+test_files[19])
predict_mask_heart = cv2.imread(test_pred+'class3/'+test_files[19])
predict_mask_left_clavicle = cv2.imread(test_pred+'class4/'+test_files[19])
predict_mask_right_clavicle = cv2.imread(test_pred+'class5/'+test_files[19])

original_image = add_colored_mask_JSRT(image, mask_image_left_lung, mask_image_right_lung, mask_image_heart, mask_image_left_clavicle, mask_image_right_clavicle)
predicted_image = add_colored_mask_JSRT(image, predict_mask_left_lung, predict_mask_right_lung, predict_mask_heart, predict_mask_left_clavicle, predict_mask_right_clavicle)


fig, axs = plt.subplots(1, 3, figsize=(15, 8))

axs[0].set_title("CT original")
axs[0].imshow(image)

axs[1].set_title("Ground truth")
axs[1].imshow(original_image)

axs[2].set_title("Predicted result")
axs[2].imshow(predicted_image)

# axs[3].set_title("diff")
# axs[3].imshow(diff_mask(mask_image, predict_mask_image))
fig.suptitle(test_files[19], fontsize=16)
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(15, 8))
axs[0][0].set_title("Right Lung")
axs[0][0].imshow(diff_mask(mask_image_right_lung, predict_mask_right_lung))
axs[0][1].set_title("Left Lung")
axs[0][1].imshow(diff_mask(mask_image_left_lung, predict_mask_left_lung))
axs[0][2].set_title("Heart")
axs[0][2].imshow(diff_mask(mask_image_heart, predict_mask_heart))
axs[1][0].set_title("Left Clavicle")
axs[1][0].imshow(diff_mask(mask_image_left_clavicle, predict_mask_left_clavicle))
axs[1][1].set_title("Right Clavicle")
axs[1][1].imshow(diff_mask(mask_image_right_clavicle, predict_mask_right_clavicle))
axs[1][2].axis("off")

fig.suptitle(test_files[19], fontsize=16)
plt.show()
