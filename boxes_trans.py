#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import cv2

train_txt = "new_train.txt"
val_txt = "new_val.txt"
new_train_txt = "resized_train.txt"
new_val_txt = "resized_val.txt"

WIDER_TRAIN1 = ""
WIDER_VAL1 = ""


def delblankline(infile, outfile):
    infopen = open(infile, 'r', encoding="utf-8")
    outfopen = open(outfile, 'w', encoding="utf-8")
    db = infopen.read()
    db_new = db.replace(' ', '\n')
    db_new = db_new.replace('/WIDER/', '/resized_WIDER/')
    outfopen.write(db_new)
    infopen.close()
    outfopen.close()

# delblankline(train_txt, new_train_txt)
# delblankline(val_txt, new_val_txt)





WIDER_ROOT = './WIDER'
# resized_root = './resized_WIDER'
train_list_file = os.path.join(WIDER_ROOT, 'wider_face_split',
                               'wider_face_train_bbx_gt.txt')
val_list_file = os.path.join(WIDER_ROOT, 'wider_face_split',
                             'wider_face_val_bbx_gt.txt')

WIDER_TRAIN = os.path.join(WIDER_ROOT, 'WIDER_train', 'images')
WIDER_VAL = os.path.join(WIDER_ROOT, 'WIDER_val', 'images')
# resized_TRAIN = os.path.join(resized_root, 'WIDER_train', 'images')
# resized_VAL = os.path.join(resized_root, 'WIDER_val', 'images')


def parse_wider_file(root, file):
    with open(file, 'r') as fr:
        lines = fr.readlines()
    face_count = []
    img_paths = []
    face_loc = []
    img_faces = []
    count = 0
    flag = False
    for k, line in enumerate(lines):
        line = line.strip().strip('\n')
        if count > 0:
            line = line.split(' ')
            count -= 1
            loc = [int(line[0]), int(line[1]), int(line[2]), int(line[3])]
            face_loc += [loc]
        if flag:
            face_count += [int(line)]
            flag = False
            count = int(line)
        if 'jpg' in line:
            img_paths += [os.path.join(root, line)]
            flag = True

    total_face = 0
    for k in face_count:
        face_ = []
        for x in range(total_face, total_face + k):
            face_.append(face_loc[x])
        img_faces += [face_]
        total_face += k
    return img_paths, img_faces


def wider_data_file():
    img_paths, bbox = parse_wider_file(WIDER_TRAIN, train_list_file)
    fw = open('train1.txt', 'w')
    for index in range(len(img_paths)):
        tmp_str = ''
        tmp_str =tmp_str+ img_paths[index]+'|'
        # path1, img_name = img_paths[index].split('/')
        path = os.path.join(img_paths[index])
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img_shape = img.shape
        boxes = bbox[index]
        print(boxes)
        for i in range(len(boxes)):
            boxes[i][0] = int(boxes[i][0]*640/img_shape[0])
            boxes[i][1] = int(boxes[i][1]*640/img_shape[1])
            boxes[i][2] = int(boxes[i][2]*640/img_shape[0])+1
            boxes[i][3] = int(boxes[i][3]*640/img_shape[1])+1
        print(boxes)

        for box in boxes:
            data = ' %d,%d,%d,%d,1'%(box[0], box[1], box[0]+box[2],  box[1]+box[3])
            tmp_str=tmp_str+data
        if len(boxes) == 0:
            print(tmp_str)
            continue
        ####err box?
        if box[2] <= 0 or box[3] <= 0:
            pass
        else:
            fw.write(tmp_str + '\n')
    fw.close()

    img_paths, bbox = parse_wider_file(WIDER_VAL, val_list_file)
    fw = open('val1.txt', 'w')
    for index in range(len(img_paths)):

        tmp_str=''
        tmp_str =tmp_str+ img_paths[index]+'|'
        path = os.path.join(img_paths[index])
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img_shape = img.shape
        boxes = bbox[index]
        for i in range(len(boxes)):
            boxes[i][0] = int(boxes[i][0]*640/img_shape[0])
            boxes[i][1] = int(boxes[i][1]*640/img_shape[1])
            boxes[i][2] = int(boxes[i][2]*640/img_shape[0])+1
            boxes[i][3] = int(boxes[i][3]*640/img_shape[1])+1
        boxes = bbox[index]

        for box in boxes:
            data = ' %d,%d,%d,%d,1'%(box[0], box[1], box[0]+box[2],  box[1]+box[3])
            tmp_str=tmp_str+data



        if len(boxes) == 0:
            print(tmp_str)
            continue
        ####err box?
        if box[2] <= 0 or box[3] <= 0:
            pass
        else:
            fw.write(tmp_str + '\n')
    fw.close()


if __name__ == '__main__':
    wider_data_file()

