#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : main.py
# @Time     : 2019/1/14 11:07 
# @Software : PyCharm
import gzip

from keras.utils import Sequence
from keras import regularizers
from keras.optimizers import Adam
from keras.engine.topology import Input
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, \
    GlobalMaxPooling2D, Lambda, MaxPooling2D, Reshape
from keras.models import Model

import threading
import os
import tensorflow as tf
import numpy as np
from random import random
from tqdm import tqdm
# Read the dataset description
from pandas import read_csv
# Determise the size of each image
from os.path import isfile
from PIL import Image as pil_image
import pickle
from imagehash import phash
from math import sqrt
# Suppress annoying stderr output when importing keras.
import keras
import random
from keras import backend as K
from keras.preprocessing.image import img_to_array, array_to_img
from scipy.ndimage import affine_transform
from keras.callbacks import ModelCheckpoint


# 指定使用显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#
# # config = tf.ConfigProto()
# # config.gpu_options.per_process_gpu_memory_fraction = 0.90  # 占用GPU90%的显存
gpu_options = tf.GPUOptions(allow_growth=True)
K.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

tagged = dict([(p, w) for _, p, w in read_csv('./data/train.csv').to_records()])
submit = [p for _, p, _ in read_csv('./data/sample_submission.csv').to_records()]
# tagged = {**tagged_playground, **tagged}

join = list(tagged.keys()) + submit

new_whale = 'new_whale'
def expand_path(p):
    if isfile('./data/train/' + p): return './data/train/' + p
    if isfile('./data/test/' + p): return './data/test/' + p
    return p


if isfile("./data/p2size.pickle"):
    with open("./data/p2size.pickle", "rb") as f:
        p2size = pickle.load(f)
else:
    p2size = {}
    for p in tqdm(join):
        size = pil_image.open(expand_path(p)).size
        p2size[p] = size
    f = open("./data/p2size.pickle", "wb")
    pickle.dump(p2size, f)

# Two phash values are considered duplicate if, for all associated image pairs:
# 1) They have the same mode and size ;
# 2) After normalizing the pixel to zero mean and variance 1.0, the mean square error does not exceed 0.1
def match(h1, h2):
    for p1 in h2ps[h1]:
        for p2 in h2ps[h2]:
            i1 = pil_image.open(expand_path(p1))
            i2 = pil_image.open(expand_path(p2))
            if i1.mode != i2.mode or i1.size != i2.size: return False
            a1 = np.array(i1)
            a1 = a1 - a1.mean()
            a1 = a1 / sqrt((a1 ** 2).mean())
            a2 = np.array(i2)
            a2 = a2 - a2.mean()
            a2 = a2 / sqrt((a2 ** 2).mean())
            a = ((a1 - a2) ** 2).mean()
            if a > 0.1: return False
    return True


if isfile('./data/p2h.pickle'):
    with open('./data/p2h.pickle', 'rb') as f:
        p2h = pickle.load(f)
else:
    print("Compute phash")
    # Compute phash for each image in the training and test set.
    p2h = {}
    for p in tqdm(join):
        img = pil_image.open(expand_path(p))
        h = phash(img)
        p2h[p] = h

    # Find all images associated with a given phash value.
    h2ps = {}
    for p, h in p2h.items():
        if h not in h2ps: h2ps[h] = []
        if p not in h2ps[h]: h2ps[h].append(p)

    # Find all distinct phash values
    hs = list(h2ps.keys())
    # If the images are close enough, associate the two phash values (this is the slow part: n^2 algorithm)
    h2h = {}
    for i, h1 in enumerate(tqdm(hs)):
        for h2 in hs[:i]:
            if h1 - h2 <= 6 and match(h1, h2):
                s1 = str(h1)
                s2 = str(h2)
                if s1 < s2: s1, s2 = s2, s1
                h2h[s1] = s2

    # Group together images with equivalent phash, and replace by string format of phash (faster and more readable)
    for p, h in p2h.items():
        h = str(h)
        if h in h2h: h = h2h[h]
        p2h[p] = h

    # 写入文件
    f = open('./data/p2h.pickle', 'wb')  # 以写模式打开二进制文件
    pickle.dump(p2h, f)


# For each image id, determine the list of pictures
h2ps = {}
for p, h in p2h.items():
    if h not in h2ps: h2ps[h] = []
    if p not in h2ps[h]: h2ps[h].append(p)


# For each images id, select the prefered image
def prefer(ps):
    if len(ps) == 1: return ps[0]
    best_p = ps[0]
    best_s = p2size[best_p]
    for i in range(1, len(ps)):
        p = ps[i]
        s = p2size[p]
        if s[0] * s[1] > best_s[0] * best_s[1]:  # Select the image with highest resolution
            best_p = p
            best_s = s
    return best_p


h2p = {}
for h, ps in h2ps.items(): h2p[h] = prefer(ps)

with open('./data/rotate.txt', 'rt') as f:
    rotate = f.read().split('\n')[:-1]
rotate = set(rotate)

# Read the bounding box data from the bounding box kernel (see reference above)
with open('./data/bounding-box.pickle', 'rb') as f:
    p2bb = pickle.load(f)

img_shape = (384, 384, 1)  # The image shape used by the model
anisotropy = 2.15  # The horizontal compression ratio
crop_margin = 0.05  # The margin added around the bounding box to compensate for bounding box inaccuracy


def build_transform(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    """
    Build a transformation matrix with the specified characteristics.
    """
    rotation = np.deg2rad(rotation)
    shear = np.deg2rad(shear)
    rotation_matrix = np.array(
        [[np.cos(rotation), np.sin(rotation), 0], [-np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
    shift_matrix = np.array([[1, 0, height_shift], [0, 1, width_shift], [0, 0, 1]])
    shear_matrix = np.array([[1, np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
    zoom_matrix = np.array([[1.0 / height_zoom, 0, 0], [0, 1.0 / width_zoom, 0], [0, 0, 1]])
    shift_matrix = np.array([[1, 0, -height_shift], [0, 1, -width_shift], [0, 0, 1]])
    return np.dot(np.dot(rotation_matrix, shear_matrix), np.dot(zoom_matrix, shift_matrix))

def read_raw_image(p):
    img = pil_image.open(expand_path(p))
    if p in rotate: img = img.rotate(180)
    return img

def read_cropped_image(p, augment):
    """
    @param p : the name of the picture to read
    @param augment: True/False if data augmentation should be performed
    @return a numpy array with the transformed image
    """
    # If an image id was given, convert to filename
    if p in h2p: p = h2p[p]
    size_x, size_y = p2size[p]

    # Determine the region of the original image we want to capture based on the bounding box.
    x0, y0, x1, y1 = p2bb[p]
    if p in rotate: x0, y0, x1, y1 = size_x - x1, size_y - y1, size_x - x0, size_y - y0
    dx = x1 - x0
    dy = y1 - y0
    x0 -= dx * crop_margin
    x1 += dx * crop_margin + 1
    y0 -= dy * crop_margin
    y1 += dy * crop_margin + 1
    if (x0 < 0): x0 = 0
    if (x1 > size_x): x1 = size_x
    if (y0 < 0): y0 = 0
    if (y1 > size_y): y1 = size_y
    dx = x1 - x0
    dy = y1 - y0
    if dx > dy * anisotropy:
        dy = 0.5 * (dx / anisotropy - dy)
        y0 -= dy
        y1 += dy
    else:
        dx = 0.5 * (dy * anisotropy - dx)
        x0 -= dx
        x1 += dx

        # Generate the transformation matrix
    trans = np.array([[1, 0, -0.5 * img_shape[0]], [0, 1, -0.5 * img_shape[1]], [0, 0, 1]])
    trans = np.dot(np.array([[(y1 - y0) / img_shape[0], 0, 0], [0, (x1 - x0) / img_shape[1], 0], [0, 0, 1]]), trans)
    if augment:
        trans = np.dot(build_transform(
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            random.uniform(0.8, 1.0),
            random.uniform(0.8, 1.0),
            random.uniform(-0.05 * (y1 - y0), 0.05 * (y1 - y0)),
            random.uniform(-0.05 * (x1 - x0), 0.05 * (x1 - x0))
        ), trans)
    trans = np.dot(np.array([[1, 0, 0.5 * (y1 + y0)], [0, 1, 0.5 * (x1 + x0)], [0, 0, 1]]), trans)

    # Read the image, transform to black and white and comvert to numpy array
    img = read_raw_image(p).convert('L')
    img = img_to_array(img)

    # Apply affine transformation
    matrix = trans[:2, :2]      # (2, 2)
    offset = trans[:2, 2]       # (2,)
    img = img.reshape(img.shape[:-1])
    # affine_transform(原图像， 旋转矩阵， 平移矩阵)  np.dot(matrix, o) + offset
    img = affine_transform(img, matrix, offset, output_shape=img_shape[:-1], order=1, mode='constant',
                           cval=np.average(img))
    img = img.reshape(img_shape)

    # Normalize to zero mean and unit variance
    img -= np.mean(img, keepdims=True)
    img /= np.std(img, keepdims=True) + K.epsilon()
    return img

def read_for_training(p):
    """
    Read and preprocess an image with data augmentation (random transform).
    """
    return read_cropped_image(p, True)


def read_for_validation(p):
    """
    Read and preprocess an image without data augmentation (use for testing).
    """
    return read_cropped_image(p, False)


def subblock(x, filter, **kwargs):
    x = BatchNormalization()(x)
    y = x
    y = Conv2D(filter, (1, 1), activation='relu', **kwargs)(y)  # Reduce the number of features to 'filter'
    y = BatchNormalization()(y)
    y = Conv2D(filter, (3, 3), activation='relu', **kwargs)(y)  # Extend the feature field
    y = BatchNormalization()(y)
    y = Conv2D(K.int_shape(x)[-1], (1, 1), **kwargs)(y)  # no activation # Restore the number of original features
    y = Add()([x, y])  # Add the bypass connection
    y = Activation('relu')(y)
    return y


def build_branch_model(lr, l2, activation='sigmoid'):
    pass

def build_model(lr, l2, activation='sigmoid'):
    ##############
    # BRANCH MODEL
    ##############
    regul = regularizers.l2(l2)
    optim = Adam(lr=lr)
    kwargs = {'padding': 'same', 'kernel_regularizer': regul}

    inp = Input(shape=img_shape)  # 384x384x1
    x = Conv2D(64, (9, 9), strides=2, activation='relu', **kwargs)(inp)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 96x96x64
    for _ in range(2):
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', **kwargs)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 48x48x64
    x = BatchNormalization()(x)
    x = Conv2D(128, (1, 1), activation='relu', **kwargs)(x)  # 48x48x128
    for _ in range(4): x = subblock(x, 64, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 24x24x128
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), activation='relu', **kwargs)(x)  # 24x24x256
    for _ in range(4): x = subblock(x, 64, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 12x12x256
    x = BatchNormalization()(x)
    x = Conv2D(384, (1, 1), activation='relu', **kwargs)(x)  # 12x12x384
    for _ in range(4): x = subblock(x, 96, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 6x6x384
    x = BatchNormalization()(x)
    x = Conv2D(512, (1, 1), activation='relu', **kwargs)(x)  # 6x6x512
    for _ in range(4): x = subblock(x, 128, **kwargs)

    x = GlobalMaxPooling2D()(x)  # 512
    branch_model = Model(inp, x)

    ############
    # HEAD MODEL
    ############
    mid = 32
    xa_inp = Input(shape=branch_model.output_shape[1:])
    xb_inp = Input(shape=branch_model.output_shape[1:])
    x1 = Lambda(lambda x: x[0] * x[1])([xa_inp, xb_inp])
    x2 = Lambda(lambda x: x[0] + x[1])([xa_inp, xb_inp])
    x3 = Lambda(lambda x: K.abs(x[0] - x[1]))([xa_inp, xb_inp])
    x4 = Lambda(lambda x: K.square(x))(x3)
    x = Concatenate()([x1, x2, x3, x4])
    x = Reshape((4, branch_model.output_shape[1], 1), name='reshape1')(x)

    # Per feature NN with shared weight is implemented using CONV2D with appropriate stride.
    x = Conv2D(mid, (4, 1), activation='relu', padding='valid')(x)
    x = Reshape((branch_model.output_shape[1], mid, 1))(x)
    x = Conv2D(1, (1, mid), activation='linear', padding='valid')(x)
    x = Flatten(name='flatten')(x)

    # Weighted sum implemented as a Dense layer.
    x = Dense(1, use_bias=True, activation=activation, name='weighted-average')(x)
    head_model = Model([xa_inp, xb_inp], x, name='head')

    ########################
    # SIAMESE NEURAL NETWORK
    ########################
    # Complete model is constructed by calling the branch model on each input image,
    # and then the head model on the resulting 512-vectors.
    img_a = Input(shape=img_shape)
    img_b = Input(shape=img_shape)
    xa = branch_model(img_a)
    xb = branch_model(img_b)
    x = head_model([xa, xb])
    model = Model([img_a, img_b], x)
    model.compile(optim, loss='binary_crossentropy', metrics=['binary_crossentropy', 'acc'])
    return model, branch_model, head_model


model, branch_model, head_model = build_model(64e-5, 2.1e-4)


with open('./data/exclude.txt', 'rt') as f:
    exclude = f.read().split('\n')[:-1]

if isfile("./data/h2ws.pickle"):
    with open("./data/h2ws.pickle", "rb") as f:
        h2ws = pickle.load(f)
else:
    # Find all the whales associated with an image id. It can be ambiguous as duplicate images may have different whale ids.
    h2ws = {}
    new_whale = 'new_whale'
    for p, w in tagged.items():
        if w != new_whale:  # Use only identified whales
            h = p2h[p]
            if h not in h2ws: h2ws[h] = []
            if w not in h2ws[h]: h2ws[h].append(w)

    for h, ws in h2ws.items():
        if len(ws) > 1:
            h2ws[h] = sorted(ws)

    f = open("./data/h2ws.pickle", "wb")
    pickle.dump(h2ws, f)


if isfile("./data/w2hs.pickle"):
    with open("./data/w2hs.pickle", "rb") as f:
        w2hs = pickle.load(f)
else:
    # For each whale, find the unambiguous images ids.
    w2hs = {}
    for h, ws in h2ws.items():
        if len(ws) == 1:  # Use only unambiguous pictures
            if h2p[h] in exclude:
                print(h)  # Skip excluded images
            else:
                w = ws[0]
                if w not in w2hs: w2hs[w] = []
                if h not in w2hs[w]: w2hs[w].append(h)
    for w, hs in w2hs.items():
        if len(hs) > 1:
            w2hs[w] = sorted(hs)

    f = open("./data/w2hs.pickle", "wb")
    pickle.dump(w2hs, f)


# Find the list of training images, keep only whales with at least two images.
train = []  # A list of training image ids
for hs in w2hs.values():
    if len(hs) > 1:
        train += hs
random.shuffle(train)
train_set = set(train)

if isfile("./data/w2ts.pickle"):
    with open("./data/w2ts.pickle", "rb") as f:
        w2ts = pickle.load(f)
else:
    w2ts = {}  # Associate the image ids from train to each whale id.
    for w, hs in w2hs.items():
        for h in hs:
            if h in train_set:
                if w not in w2ts: w2ts[w] = []
                if h not in w2ts[w]: w2ts[w].append(h)
    f = open("./data/w2ts.pickle", "wb")
    pickle.dump(w2ts, f)

for w, ts in w2ts.items(): w2ts[w] = np.array(ts)


t2i = {}  # The position in train of each training image id
for i, t in enumerate(train): t2i[t] = i

# First try to use lapjv Linear Assignment Problem solver as it is much faster.
# At the time I am writing this, kaggle kernel with custom package fail to commit.
# scipy can be used as a fallback, but it is too slow to run this kernel under the time limit
# As a workaround, use scipy with data partitioning.
# Because algorithm is O(n^3), small partitions are much faster, but not what produced the submitted solution
try:
    # from lap import lapjv
    from lapjv import lapjv
    segment = False
except ImportError:
    print('Module lap not found, emulating with much slower scipy.optimize.linear_sum_assignment')
    segment = True
    from scipy.optimize import linear_sum_assignment


class TrainingData(Sequence):
    def __init__(self, score, steps=1000, batch_size=32):
        """
        @param score the cost matrix for the picture matching
        @param steps the number of epoch we are planning with this score matrix
        """
        super(TrainingData, self).__init__()
        self.score = -score  # Maximizing the score is the same as minimuzing -score.
        self.steps = steps
        self.batch_size = batch_size
        for ts in w2ts.values():
            idxs = [t2i[t] for t in ts]
            for i in idxs:
                for j in idxs:
                    # Set a large value for matching whales -- eliminates this potential pairing
                    self.score[i, j] = 10000.0
        self.on_epoch_end()

    def __getitem__(self, index):
        start = self.batch_size * index
        end = min(start + self.batch_size, len(self.match) + len(self.unmatch))
        size = end - start
        assert size > 0
        a = np.zeros((size,) + img_shape, dtype=K.floatx())
        b = np.zeros((size,) + img_shape, dtype=K.floatx())
        c = np.zeros((size, 1), dtype=K.floatx())
        j = start // 2
        for i in range(0, size, 2):
            a[i, :, :, :] = read_for_training(self.match[j][0])
            b[i, :, :, :] = read_for_training(self.match[j][1])
            c[i, 0] = 1  # This is a match
            a[i + 1, :, :, :] = read_for_training(self.unmatch[j][0])
            b[i + 1, :, :, :] = read_for_training(self.unmatch[j][1])
            c[i + 1, 0] = 0  # Different whales
            j += 1
        return [a, b], c

    def compute_LAP(self):
        num_threads = 6
        tmp = num_threads * [None]
        threads = []
        thread_input = num_threads * [None]
        thread_idx = 0
        batch = self.score.shape[0] // (num_threads - 1)
        for start in range(0, self.score.shape[0], batch):
            end = min(self.score.shape[0], start + batch)
            thread_input[thread_idx] = self.score[start:end, start:end]
            thread_idx += 1

        def worker(data_idx):
            x, _, _ = lapjv(thread_input[data_idx])
            tmp[data_idx] = x

        # print("Start worker threads")
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            if t is not None:
                t.join()
        x = np.concatenate(tmp)
        # print("LAP completed")
        return x

    def on_epoch_end(self):
        if self.steps <= 0: return  # Skip this on the last epoch.
        self.steps -= 1
        self.match = []
        self.unmatch = []

        # compute x
        # _, _, x = lapjv(self.score)  # Solve the linear assignment problem
        x = self.compute_LAP()


        y = np.arange(len(x), dtype=np.int32)

        # Compute a derangement for matching whales
        for ts in w2ts.values():
            d = ts.copy()
            while True:
                random.shuffle(d)
                if not np.any(ts == d): break
            for ab in zip(ts, d): self.match.append(ab)

        # Construct unmatched whale pairs from the LAP solution.
        for i, j in zip(x, y):
            if i == j:
                print(self.score)
                print(x)
                print(y)
                print(i, j)
            assert i != j
            self.unmatch.append((train[i], train[j]))

        # Force a different choice for an eventual next epoch.
        self.score[x, y] = 10000.0
        self.score[y, x] = 10000.0
        random.shuffle(self.match)
        random.shuffle(self.unmatch)
        # print(len(self.match), len(train), len(self.unmatch), len(train))
        assert len(self.match) == len(train) and len(self.unmatch) == len(train)

    def __len__(self):
        return (len(self.match) + len(self.unmatch) + self.batch_size - 1) // self.batch_size


# Test on a batch of 32 with random costs.
# score = np.random.random_sample(size=(len(train), len(train)))
# data = TrainingData(score)
# (a, b), c = data[0]

# A Keras generator to evaluate only the BRANCH MODEL
class FeatureGen(Sequence):
    def __init__(self, data, batch_size=64, verbose=1):
        super(FeatureGen, self).__init__()
        self.data       = data
        self.batch_size = batch_size
        self.verbose    = verbose
        if self.verbose > 0: self.progress = tqdm(total=len(self), desc='Features')
    def __getitem__(self, index):
        start = self.batch_size*index
        size  = min(len(self.data) - start, self.batch_size)
        a     = np.zeros((size,) + img_shape, dtype=K.floatx())
        for i in range(size): a[i,:,:,:] = read_for_validation(self.data[start + i])
        if self.verbose > 0:
            self.progress.update()
            if self.progress.n >= len(self): self.progress.close()
        return a
    def __len__(self):
        return (len(self.data) + self.batch_size - 1)//self.batch_size

# A Keras generator to evaluate on the HEAD MODEL on features already pre-computed.
# It computes only the upper triangular matrix of the cost matrix if y is None.
class ScoreGen(Sequence):
    def __init__(self, x, y=None, batch_size=2048, verbose=1):
        super(ScoreGen, self).__init__()
        self.x          = x
        self.y          = y
        self.batch_size = batch_size
        self.verbose    = verbose
        if y is None:
            self.y           = self.x
            self.ix, self.iy = np.triu_indices(x.shape[0],1)
        else:
            self.iy, self.ix = np.indices((y.shape[0],x.shape[0]))
            self.ix          = self.ix.reshape((self.ix.size,))
            self.iy          = self.iy.reshape((self.iy.size,))
        self.subbatch = (len(self.x) + self.batch_size - 1)//self.batch_size
        if self.verbose > 0: self.progress = tqdm(total=len(self), desc='Scores')
    def __getitem__(self, index):
        start = index*self.batch_size
        end   = min(start + self.batch_size, len(self.ix))
        a     = self.y[self.iy[start:end],:]
        b     = self.x[self.ix[start:end],:]
        if self.verbose > 0:
            self.progress.update()
            if self.progress.n >= len(self): self.progress.close()
        return [a,b]
    def __len__(self):
        return (len(self.ix) + self.batch_size - 1)//self.batch_size


def set_lr(model, lr):
    K.set_value(model.optimizer.lr, float(lr))


def get_lr(model):
    return K.get_value(model.optimizer.lr)


def score_reshape(score, x, y=None):
    """
    Tranformed the packed matrix 'score' into a square matrix.
    @param score the packed matrix
    @param x the first image feature tensor
    @param y the second image feature tensor if different from x
    @result the square matrix
    """
    if y is None:
        # When y is None, score is a packed upper triangular matrix.
        # Unpack, and transpose to form the symmetrical lower triangular matrix.
        m = np.zeros((x.shape[0], x.shape[0]), dtype=K.floatx())
        m[np.triu_indices(x.shape[0], 1)] = score.squeeze()
        m += m.transpose()
    else:
        m = np.zeros((y.shape[0], x.shape[0]), dtype=K.floatx())
        iy, ix = np.indices((y.shape[0], x.shape[0]))
        ix = ix.reshape((ix.size,))
        iy = iy.reshape((iy.size,))
        m[iy, ix] = score.squeeze()
    return m

def compute_score(verbose=1):
    """
    Compute the score matrix by scoring every pictures from the training set against every other picture O(n^2).
    """
    features = branch_model.predict_generator(FeatureGen(train, verbose=verbose), max_queue_size=12, workers=6,
                                              verbose=0)
    score = head_model.predict_generator(ScoreGen(features, verbose=verbose), max_queue_size=12, workers=6,
                                         verbose=0)
    score = score_reshape(score, features)
    return features, score

checkpoint = ModelCheckpoint(filepath="./data/model_jky/checkpoint-{epoch:05d}-{loss:.3f}.h5",
                             monitor='loss',mode='min', period=1, verbose=2)


def make_steps(step, ampl):
    """
    Perform training epochs
    @param step Number of epochs to perform
    @param ampl the K, the randomized component of the score matrix.
    """
    global w2ts, t2i, steps, features, score, histories

    # shuffle the training pictures
    random.shuffle(train)
    # Map whale id to the list of associated training picture hash value
    with open("./data/w2ts.pickle", "rb") as f:
        w2ts = pickle.load(f)
    for w, ts in w2ts.items(): w2ts[w] = np.array(ts)

    # Map training picture hash value to index in 'train' array
    t2i = {}
    for i, t in enumerate(train): t2i[t] = i

    # Compute the match score for each picture pair
    features, score = compute_score()

    # Train the model for 'step' epochs
    history = model.fit_generator(
        TrainingData(score + ampl * np.random.random_sample(size=score.shape), steps=step, batch_size=32),
        initial_epoch=steps, epochs=steps + step, max_queue_size=12, workers=6, verbose=2,
        callbacks=[checkpoint]
        # callbacks=[
        #     TQDMNotebookCallback(leave_inner=True, metric_format='{value:0.3f}')
        # ]
    ).history
    steps += step

    # # Collect history data
    # history['epochs'] = steps
    # history['ms'] = np.mean(score)
    # history['lr'] = get_lr(model)
    # print(history['epochs'], history['lr'], history['ms'])
    # histories.append(history)

model_name = 'mpiotte-standard'
histories = []
steps = 600

if isfile('./data/model_jky/checkpoint-00605-0.022.h5'):
    tmp = keras.models.load_model('./data/model_jky/checkpoint-00605-0.022.h5')
    model.set_weights(tmp.get_weights())
    print("load weight")
else:
    print("load fail")

# epoch -> 10
# make_steps(10, 1000)
# ampl = 100.0
# for _ in range(10):
#     print('noise ampl.  = ', ampl)
# make_steps(5, 30)
# # make_steps(2, 1.0)
# ampl = max(1.0, 100 ** -0.1 * ampl)
# # epoch -> 150
# for _ in range(18): make_steps(5, 1.0)
# # epoch -> 200
# set_lr(model, 16e-5)
# for _ in range(10): make_steps(5, 0.5)
# # epoch -> 240
# set_lr(model, 4e-5)
# for _ in range(8): make_steps(5, 0.25)
# epoch -> 250
# set_lr(model, 1e-5)
# for _ in range(2): make_steps(5, 0.25)
# # epoch -> 300
# weights = model.get_weights()
# model, branch_model, head_model = build_model(64e-5, 0.0002)
# model.set_weights(weights)
# for _ in range(10): make_steps(5, 1.0)
# # epoch -> 350
# set_lr(model, 16e-5)
# for _ in range(10): make_steps(5, 0.5)
# # epoch -> 390
# set_lr(model, 4e-5)
# for _ in range(8): make_steps(5, 0.25)
# epoch -> 400

# epoch -> 550
# set_lr(model, 1e-5)
# make_steps(5, 1000)
# make_steps(5, 100)
# make_steps(5, 1)
# make_steps(5, 0.5)
# for _ in range(10): make_steps(5, 0.20)
# set_lr(model, 5e-6)
# for _ in range(10): make_steps(5, 0.15)

set_lr(model, 1e-5)
for _ in range(10): make_steps(5, 0.05)
for _ in range(10): make_steps(5, 0.04)
for _ in range(10): make_steps(5, 0.02)
for _ in range(10): make_steps(5, 0.005)
model.save('mpiotte-standard.model')
# train()
# predict()
# nohup python3 -u  main.py   > logs.out 2>&1 &
# 34336