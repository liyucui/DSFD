import os
import cv2
import numpy as np

img_dir = "/home/lucy/PycharmProjects/DSFD-tensorflow-master/WIDER/WIDER_val/images"
dst_dir = "/home/lucy/PycharmProjects/DSFD-tensorflow-master/resized_WIDER/WIDER_val/images"

# dst_val_dir = "/home/lucy/PycharmProjects/human_detection/resized/val/negative/"

if os.path.exists(dst_dir) is False:
    os.makedirs(dst_dir)
# if os.path.exists(dst_val_dir) is False:
#     os.makedirs(dst_val_dir)

dst_size = (640, 640)
img_dirs = os.listdir(img_dir)
print(img_dirs)
length_dirs = len(img_dirs)
length_dir = len(img_dir)
total_img_count = length_dirs - length_dir
image_count = 0

for dir in img_dirs:
    img_paths = os.path.join(img_dir, dir)
    img_dst_path = os.path.join(dst_dir, dir)
    if os.path.exists(img_dst_path) is False:
        os.makedirs(img_dst_path)
    image = os.listdir(img_paths)
    for img in image:
        image_path = os.path.join(img_paths, img)
        img_names = img
        img_array = cv2.imread(image_path, cv2.IMREAD_COLOR)
        new_array = cv2.resize(img_array, dst_size, interpolation=cv2.INTER_AREA)
        image_count += 1
        new_img_name = str(img_names)+'.jpg'

        save_img = img_dst_path +'/'+new_img_name
        cv2.imwrite(save_img, new_array)
        print('%d/%d' %(image_count, total_img_count))










