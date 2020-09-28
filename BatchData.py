import os
import numpy as np
import cv2
import imageio

class BatchData:
    train_img_list = []
    train_label_list = []
    val_img_list = []
    val_label_list = []
    batch_offset = 0
    epochs_completed = 0
    train_index = []
    train_count = 0
    train_label = float(-1)
    val_label = float(-1)
    def __init__(self, data_dir):
        for file in os.listdir(data_dir+'/train'):
            for files in os.listdir(data_dir+'/train/'+file):
                self.train_count += 1
                self.train_img_list.append(data_dir+'/train/'+file +'/'+files)
                if file == 'positive':
                    self.train_label = float(1)
                    self.train_label_list.append(self.train_label)
                else:
                    self.train_label = float(-1)
                    self.train_label_list.append(self.train_label)
            # print(len(files))
        self.train_index = np.arange(self.train_count)
        for file in os.listdir(data_dir+'/val'):
            for files in os.listdir(data_dir+'/val/'+file):
                self.val_img_list.append(data_dir+'/val/'+file+'/'+files)
                if file == 'positive':
                    self.val_label = float(1)
                    self.val_label_list.append(self.val_label)
                else:
                    self.val_label = float(-1)
                    self.val_label_list.append(self.val_label)



    def train_next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        perm = np.arange(len(self.train_img_list))
        np.random.shuffle(perm)
        self.train_index = np.array(self.train_index)[perm]

        if self.batch_offset > len(self.train_img_list):
            self.epochs_completed += 1
            print("*****Epochs completed: " + str(self.epochs_completed) + "******")
            perm = np.arange(len(self.train_img_list))
            np.random.shuffle(perm)
            self.train_index = np.array(self.train_index)[perm]

            start = 0
            self.batch_offset = batch_size
        end = self.batch_offset

        imgdata = []
        matdata = []

        for n in self.train_index[start:end]:
            try:
                imgdata.append(imageio.imread(self.train_img_list[n]))
                matdata.append(self.train_label_list[n])
            except ValueError:
                if len(np.shape(imgdata)) > len(np.shape(matdata)):
                    imgdata.pop()
                elif len(np.shape(imgdata)) < len(np.shape(matdata)):
                    matdata.pop()
                else:
                    print('FileNotFoundError: %s, %s' % self.train_img_list[n], self.train_label_list[n])
                end += 1
                self.batch_offset += 1
                if self.batch_offset > len(self.train_img_list):
                    self.epochs_completed += 1
                    print("*****Epochs completed: " + str(self.epochs_completed) + "******")
                    start = 0
                    self.batch_offset = batch_size
        imgdata = np.array(imgdata, dtype=np.float32)
        matdata = np.array(matdata, dtype=np.float32)
        matdata = np.expand_dims(matdata, 1)
        return imgdata, matdata

    def val_random_batch(self, batch_size):
        indices = np.random.randint(0, len(self.val_img_list), size=[batch_size]).tolist()

        imgdata = []
        matdata = []
        for n in indices:
            try:
                imgdata.append(imageio.imread(self.val_img_list[n]))
                matdata.append(self.val_label_list[n])
            except ValueError:
                if len(np.shape(imgdata)) > len(np.shape(matdata)):
                    imgdata.pop()
                elif len(np.shape(imgdata)) < len(np.shape(matdata)):
                    matdata.pop()
                else:
                    print('FileNotFoundError: %s, %s' % self.train_img_list[n], self.train_label_list[n])

        imgdata = np.array(imgdata, dtype=np.float32)
        matdata = np.array(matdata, dtype=np.float32)
        matdata = np.expand_dims(matdata, 1)
        return imgdata, matdata







