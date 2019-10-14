import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import keras
import glob
from random import sample, randint, shuffle
from keras.utils import Sequence, to_categorical
import cv2
seed = 8
np.random.seed(seed)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, paths, graph_conv_filters, timesteps, mode, num_classes, stack_size, batch_size=32):
        'Initialization'
        self.batch_size = batch_size
        self.path_skeleton = paths['skeleton']
        self.path_cnn = paths['cnn']
        self.files = [i.strip() for i in open(paths['split_path'] + mode + '.txt').readlines()]
        self.graph_conv_filters = graph_conv_filters
        self.num_classes = num_classes
        self.stack_size = stack_size
        self.stride = 2
        self.step = timesteps
        self.dim = 150  ##for two skeletons in a single frame
        self.mode = mode
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.files) / self.batch_size)

    def __getitem__(self, idx):

        batch = self.files[idx * self.batch_size: (idx + 1) * self.batch_size]
        graph_conv = self.graph_conv_filters[0:self.batch_size]
        x_data_cnn = self._get_data_cnn(batch)
        X = self._get_data_skeleton(batch)

        y_data = np.array([int(i[-3:]) for i in batch]) - 1
        y_data = to_categorical(y_data, num_classes=self.num_classes)
        y_reg = np.zeros([self.batch_size])
        #y_reg = to_categorical(y_reg, num_classes=1)

        return [X[:, 0, :, :], X[:, 1, :, :], X[:, 2, :, :], X[:, 3, :, :], X[:, 4, :, :], X[:, 5, :, :], X[:, 6, :, :],
                X[:, 7, :, :], X[:, 8, :, :], X[:, 9, :, :], X[:, 10, :, :], X[:, 11, :, :], X[:, 12, :, :], X[:, 13, :, :],
                X[:, 14, :, :], X[:, 15, :, :], X, graph_conv, x_data_cnn], [y_data, y_reg]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.files))
        if self.mode == 'train' or self.mode == 'train_set': 
            shuffle(self.files)

    def _get_data_skeleton(self, list_IDs_temp):

        # Initialization
        X = np.empty((self.batch_size, self.step, self.dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            unpadded_file = np.load(self.path_skeleton + ID + '.npy')
            origin = unpadded_file[0, 3:6]
            [row, col] = unpadded_file.shape
            origin = np.tile(origin, (row, 50))
            unpadded_file -= origin
            extra_frames = (len(unpadded_file) % self.step)
            if extra_frames < (self.step / 2):
                padded_file = unpadded_file[0:len(unpadded_file) - extra_frames, :]
            else:
                [row, col] = unpadded_file.shape
                alpha = int(len(unpadded_file) / self.step) + 1
                req_pad = np.zeros(((alpha * self.step) - row, col))
                padded_file = np.vstack((unpadded_file, req_pad))
            splitted_file = np.split(padded_file, self.step)
            splitted_file = np.asarray(splitted_file)
            row, col, width = splitted_file.shape
            sampled_file = []
            for k in range(0, self.step):
                c = np.random.choice(col, 1)
                sampled_file.append(splitted_file[k, c, :])
            sampled_file = np.asarray(sampled_file)
            X[i,] = np.squeeze(sampled_file)

        X = X[:,:, 0:75]
        X = np.reshape(X, [self.batch_size, self.step, 25, 3])

        return X

    def _get_data_cnn(self, batch):

        x_train = [self._get_video(i) for i in batch]
        x_train = np.array(x_train, np.float32)
        x_train /= 127.5
        x_train -= 1

        return x_train

    def _get_video(self, vid_name):
        images = glob.glob(self.path_cnn + vid_name + "/*")

        images.sort()
        files = []
        if len(images) > (self.stack_size * self.stride):
            start = randint(0, len(images) - self.stack_size * self.stride)
            files.extend([images[i] for i in range(start, (start + self.stack_size * self.stride), self.stride)])
        elif len(images) < self.stack_size:
            files.extend(images)
            while len(files) < self.stack_size:
                files.extend(images)
            files = files[:self.stack_size]
        else:
            start = randint(0, len(images) - self.stack_size)
            files.extend([images[i] for i in range(start, (start + self.stack_size))])

        files.sort()

        arr = []
        for i in files:
            if os.path.isfile(i):
                arr.append(cv2.resize(cv2.imread(i), (224, 224)))
            else:
                arr.append(arr[-1])

        return arr
