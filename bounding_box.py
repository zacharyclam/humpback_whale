#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : bounding_box.py
# @Time     : 2019/1/20 19:41 
# @Software : PyCharm
import random
from os.path import isfile

import numpy as np
from PIL import Image as pil_image
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.engine.topology import Input
from keras.layers import BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.utils import Sequence
from numpy.linalg import inv as mat_inv
from scipy.ndimage import affine_transform
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf
import os

# 指定使用显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.90  # 占用GPU90%的显存
K.set_session(tf.Session(config=config))

# Define useful constants

img_shape = (128, 128, 1)
anisotropy = 2.15

with open('./data2/cropping.txt', 'rt') as f:
    data = f.read().split('\n')[:-1]

data = [line.split(',') for line in data]
data = [(p, [(int(coord[i]), int(coord[i + 1])) for i in range(0, len(coord), 2)]) for p, *coord in data]


def expand_path(p):
    if isfile('./data/train/' + p): return './data/train/' + p
    if isfile('./data/test/' + p): return './data/test/' + p
    if isfile('./data2/train/' + p): return './data2/train/' + p
    if isfile('./data2/test/' + p): return './data2/test/' + p
    return p


def read_raw_image(p):
    return pil_image.open(expand_path(p))


def draw_dot(draw, x, y):
    draw.ellipse(((x - 5, y - 5), (x + 5, y + 5)), fill='red', outline='red')


def draw_dots(draw, coordinates):
    for x, y in coordinates: draw_dot(draw, x, y)


def bounding_rectangle(list):
    x0, y0 = list[0]
    x1, y1 = x0, y0
    for x, y in list[1:]:
        x0 = min(x0, x)
        y0 = min(y0, y)
        x1 = max(x1, x)
        y1 = max(y1, y)
    return x0, y0, x1, y1


# Read an image as black&white numpy array
def read_array(p):
    img = read_raw_image(p).convert('L')
    return img_to_array(img)


def build_transform(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    rotation = np.deg2rad(rotation)
    shear = np.deg2rad(shear)
    rotation_matrix = np.array(
        [[np.cos(rotation), np.sin(rotation), 0], [-np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
    shift_matrix = np.array([[1, 0, height_shift], [0, 1, width_shift], [0, 0, 1]])
    shear_matrix = np.array([[1, np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
    zoom_matrix = np.array([[1.0 / height_zoom, 0, 0], [0, 1.0 / width_zoom, 0], [0, 0, 1]])
    shift_matrix = np.array([[1, 0, -height_shift], [0, 1, -width_shift], [0, 0, 1]])
    return np.dot(np.dot(rotation_matrix, shear_matrix), np.dot(zoom_matrix, shift_matrix))


# Compute the coordinate transformation required to center the pictures, padding as required.
def center_transform(affine, input_shape):
    hi, wi = float(input_shape[0]), float(input_shape[1])
    ho, wo = float(img_shape[0]), float(img_shape[1])
    top, left, bottom, right = 0, 0, hi, wi
    if wi / hi / anisotropy < wo / ho:  # input image too narrow, extend width
        w = hi * wo / ho * anisotropy
        left = (wi - w) / 2
        right = left + w
    else:  # input image too wide, extend height
        h = wi * ho / wo / anisotropy
        top = (hi - h) / 2
        bottom = top + h
    center_matrix = np.array([[1, 0, -ho / 2], [0, 1, -wo / 2], [0, 0, 1]])
    scale_matrix = np.array([[(bottom - top) / ho, 0, 0], [0, (right - left) / wo, 0], [0, 0, 1]])
    decenter_matrix = np.array([[1, 0, hi / 2], [0, 1, wi / 2], [0, 0, 1]])
    return np.dot(np.dot(decenter_matrix, scale_matrix), np.dot(affine, center_matrix))


# Apply an affine transformation to an image represented as a numpy array.
def transform_img(x, affine):
    matrix = affine[:2, :2]
    offset = affine[:2, 2]
    x = np.moveaxis(x, -1, 0)
    channels = [affine_transform(channel, matrix, offset, output_shape=img_shape[:-1], order=1,
                                 mode='constant', cval=np.average(channel)) for channel in x]
    return np.moveaxis(np.stack(channels, axis=0), 0, -1)


# Read an image for validation, i.e. without data augmentation.
def read_for_validation(p):
    x = read_array(p)
    t = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    t = center_transform(t, x.shape)
    x = transform_img(x, t)
    x -= np.mean(x, keepdims=True)
    x /= np.std(x, keepdims=True) + K.epsilon()
    return x, t


# Read an image for training, i.e. including a random affine transformation
def read_for_training(p):
    x = read_array(p)
    t = build_transform(
        random.uniform(-5, 5),
        random.uniform(-5, 5),
        random.uniform(0.9, 1.0),
        random.uniform(0.9, 1.0),
        random.uniform(-0.05 * img_shape[0], 0.05 * img_shape[0]),
        random.uniform(-0.05 * img_shape[1], 0.05 * img_shape[1]))
    t = center_transform(t, x.shape)
    x = transform_img(x, t)
    x -= np.mean(x, keepdims=True)
    x /= np.std(x, keepdims=True) + K.epsilon()
    return x, t


# Transform corrdinates according to the provided affine transformation
def coord_transform(list, trans):
    result = []
    for x, y in list:
        y, x, _ = trans.dot([y, x, 1]).astype(np.int)
        result.append((x, y))
    return result


train, val = train_test_split(data, test_size=200, random_state=1)
train += train
train += train
train += train
train += train


class TrainingData(Sequence):
    def __init__(self, batch_size=32):
        super(TrainingData, self).__init__()
        self.batch_size = batch_size

    def __getitem__(self, index):
        start = self.batch_size * index
        end = min(len(train), start + self.batch_size)
        size = end - start
        a = np.zeros((size,) + img_shape, dtype=K.floatx())
        b = np.zeros((size, 4), dtype=K.floatx())
        for i, (p, coords) in enumerate(train[start:end]):
            img, trans = read_for_training(p)
            coords = coord_transform(coords, mat_inv(trans))
            x0, y0, x1, y1 = bounding_rectangle(coords)
            a[i, :, :, :] = img
            b[i, 0] = x0
            b[i, 1] = y0
            b[i, 2] = x1
            b[i, 3] = y1
        return a, b

    def __len__(self):
        return (len(train) + self.batch_size - 1) // self.batch_size


val_a = np.zeros((len(val),) + img_shape, dtype=K.floatx())  # Preprocess validation images
val_b = np.zeros((len(val), 4), dtype=K.floatx())  # Preprocess bounding boxes
for i, (p, coords) in enumerate(tqdm(val)):
    try:
        img, trans = read_for_validation(p)
    except FileNotFoundError:
        print("file not found")
        continue
    coords = coord_transform(coords, mat_inv(trans))
    x0, y0, x1, y1 = bounding_rectangle(coords)
    val_a[i, :, :, :] = img
    val_b[i, 0] = x0
    val_b[i, 1] = y0
    val_b[i, 2] = x1
    val_b[i, 3] = y1


def build_model(with_dropout=True):
    kwargs = {'activation': 'relu', 'padding': 'same'}
    conv_drop = 0.2
    dense_drop = 0.5
    inp = Input(shape=img_shape)

    x = inp

    x = Conv2D(64, (9, 9), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    h = MaxPooling2D(pool_size=(1, int(x.shape[2])))(x)
    h = Flatten()(h)
    if with_dropout: h = Dropout(dense_drop)(h)
    h = Dense(16, activation='relu')(h)

    v = MaxPooling2D(pool_size=(int(x.shape[1]), 1))(x)
    v = Flatten()(v)
    if with_dropout: v = Dropout(dense_drop)(v)
    v = Dense(16, activation='relu')(v)

    x = Concatenate()([h, v])
    if with_dropout: x = Dropout(0.5)(x)
    x = Dense(4, activation='linear')(x)
    return Model(inp, x)


model = build_model(with_dropout=True)
model.compile(Adam(lr=0.032), loss='mean_squared_error')
# for num in range(1, 4):
#     model_name = 'cropping-%01d.h5' % num
#     print(model_name)
#     model.compile(Adam(lr=0.032), loss='mean_squared_error')
#     model.fit_generator(
#         TrainingData(), epochs=70, max_queue_size=12, workers=4, verbose=2,
#         validation_data=(val_a, val_b),
#         callbacks=[
#             EarlyStopping(monitor='val_loss', patience=9, min_delta=0.1, verbose=1),
#             ReduceLROnPlateau(monitor='val_loss', patience=3, min_delta=0.1, factor=0.25, min_lr=0.002, verbose=1),
#             ModelCheckpoint(model_name, save_best_only=True, save_weights_only=True),
#         ])
#     model.load_weights(model_name)
#     model.evaluate(val_a, val_b, verbose=0)
#
model.load_weights('cropping-1.h5')
loss1 = model.evaluate(val_a, val_b, verbose=0)
model.load_weights('cropping-2.h5')
loss2 = model.evaluate(val_a, val_b, verbose=0)
model.load_weights('cropping-3.h5')
loss3 = model.evaluate(val_a, val_b, verbose=0)
model_name = 'cropping-1.h5'
if loss2 <= loss1 and loss2 < loss3: model_name = 'cropping-2.h5'
if loss3 <= loss1 and loss3 <= loss2: model_name = 'cropping-3.h5'
model.load_weights(model_name)

model2 = build_model(with_dropout=False)
model2.load_weights(model_name)

model2.compile(Adam(lr=0.002), loss='mean_squared_error')
model2.evaluate(val_a, val_b, verbose=0)
#
# # Recompute the mean and variance running average without dropout
for layer in model2.layers:
    if not isinstance(layer, BatchNormalization):
        layer.trainable = False
model2.compile(Adam(lr=0.002), loss='mean_squared_error')
model2.fit_generator(TrainingData(), epochs=1, max_queue_size=12, workers=6, verbose=1, validation_data=(val_a, val_b))
for layer in model2.layers:
    if not isinstance(layer, BatchNormalization):
        layer.trainable = True
model2.compile(Adam(lr=0.002), loss='mean_squared_error')
model2.save('cropping.model')

# Generate best bounding boxes
from pandas import read_csv

tagged = [p for _, p, _ in read_csv('./data2/train.csv').to_records()] +  \
         [p for _, p, _ in read_csv('./data/train.csv').to_records()]
submit = [p for _, p, _ in read_csv('./data2/sample_submission.csv').to_records()] + \
         [p for _, p, _ in read_csv('./data/sample_submission.csv').to_records()]
join = tagged + submit
# If the picture is part of the bounding box dataset, use the golden value.
p2bb = {}
for i, (p, coords) in enumerate(data):
    p2bb[p] = bounding_rectangle(coords)
# For other pictures, evaluate the model.
p2bb = {}
for p in tqdm(join):
    if p not in p2bb:
        img, trans = read_for_validation(p)
        a = np.expand_dims(img, axis=0)
        x0, y0, x1, y1 = model2.predict(a).squeeze()
        (u0, v0), (u1, v1) = coord_transform([(x0, y0), (x1, y1)], trans)
        p2bb[p] = (u0, v0, u1, v1)
import pickle

with open('bounding-box.pickle', 'wb') as f:
    pickle.dump(p2bb, f)