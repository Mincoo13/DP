import tensorflow
import keras
import numpy as np
import pandas
import sklearn
import sys
import os
import cv2
import matplotlib.pyplot as plt

import pixellib
from pixellib.semantic import semantic_segmentation
from pixellib.instance import instance_segmentation

image = cv2.imread(r"..\dp_dataset\lemna_all_cropped\80C (2).jpg")
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # plt.imshow(img)
    # plt.show()

img2 = cv2.imread(r"..\dp_dataset\lemna_all_cropped\L 10C.jpg")



# remove noise
img2 = cv2.GaussianBlur(image,(3,3),0)
# plt.imshow(img2)
# plt.show()

while True:

    dim = (300, 300)
    img = cv2.resize(img2, dim, interpolation=cv2.INTER_NEAREST)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower_green = np.array([35, 80, 50])
    # upper_green = np.array([80, 255, 200])

    lower_green = np.array([35, 80, 50])
    upper_green = np.array([80, 255, 200])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    res = cv2.bitwise_and(img, img, mask=mask)

    kernel = np.ones((5, 5), np.uint8)
    dilatation = cv2.dilate(mask, kernel, iterations=1)
    # plt.imshow(res)
    # plt.show()
    cv2.imshow('dilate', res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

sys.exit()
# https://towardsdatascience.com/image-segmentation-with-six-lines-0f-code-acb870a462e8
segment_image_semantic = semantic_segmentation()
segment_image_semantic.load_pascalvoc_model(r"models\deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
segment_image_semantic.segmentAsPascalvoc(r"..\dp_dataset\lemna_all_cropped\L 10C.jpg",
                                 output_image_name="segmented_semantic.jpg")
segment_image_semantic.segmentAsPascalvoc(r"..\dp_dataset\lemna_all_cropped\L 10C.jpg",
                                 output_image_name="segmented_semantic_overlay.jpg", overlay=True)

segment_image_instance = instance_segmentation()
segment_image_instance.load_model(r"models\mask_rcnn_coco.h5")
segment_image_instance.segmentImage(r"..\dp_dataset\lemna_all_cropped\L 10C.jpg",
                                     output_image_name="segmented_instance.jpg", show_bboxes=True)

sys.exit()

print(os.getcwd())
path = '..\dp_dataset\lemna_all_cropped'
folder_data = os.listdir(path)
x_data = []
for image_path in folder_data:
    # print(path + '\\' + image_path)
    image = cv2.imread(path + '\\' + image_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dim = (300, 300)
    image_resized = cv2.resize(image, dim, interpolation=cv2.INTER_NEAREST)
    x_data.append(np.array(image_resized))

plt.imshow(x_data[0])
plt.show()
print(x_data[0])
