#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : triplet_loss.py
# @Time     : 2019/1/31 16:40 
# @Software : PyCharm

from collections import defaultdict
from os.path import join

import numpy as np
import pandas as pd
from PIL import Image
from keras import backend as K
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Input, Dense, Dropout, Lambda
from keras.models import Model
from keras.optimizers import Adam
from sklearn.cross_validation import train_test_split

import os
import tensorflow  as tf
# 指定使用显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.90  # 占用GPU90%的显存
gpu_options = tf.GPUOptions(allow_growth=True)
K.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# define parameter
batch_size = 24
embedding_dim = 50
image_size = 224
path_base = './data/'
path_train = join(path_base,'train')
path_test = join(path_base,'test')

path_model = join(path_base,'MyModel.hdf5')
path_csv = './data/train.csv'


class sample_gen(object):
    def __init__(self, file_class_mapping, other_class="new_whale"):
        self.file_class_mapping = file_class_mapping
        self.class_to_list_files = defaultdict(list)
        self.list_other_class = []
        self.list_all_files = list(file_class_mapping.keys())
        self.range_all_files = list(range(len(self.list_all_files)))

        for file, class_ in file_class_mapping.items():
            if class_ == other_class:
                self.list_other_class.append(file)
            else:
                self.class_to_list_files[class_].append(file)

        self.list_classes = list(set(self.file_class_mapping.values()))
        self.range_list_classes = range(len(self.list_classes))
        self.class_weight = np.array([len(self.class_to_list_files[class_]) for class_ in self.list_classes])
        self.class_weight = self.class_weight / np.sum(self.class_weight)

    def get_sample(self):
        # 从self.range_list_classes 以概率p随机选择1个 class
        class_idx = np.random.choice(self.range_list_classes, 1, p=self.class_weight)[0]
        # 从该class对应的鲸鱼图片中随机选择两个
        examples_class_idx = np.random.choice(range(len(self.class_to_list_files[self.list_classes[class_idx]])), 2)
        positive_example_1, positive_example_2 = \
            self.class_to_list_files[self.list_classes[class_idx]][examples_class_idx[0]], \
            self.class_to_list_files[self.list_classes[class_idx]][examples_class_idx[1]]

        negative_example = None
        while negative_example is None or self.file_class_mapping[negative_example] == \
                self.file_class_mapping[positive_example_1]:
            # 从所有样本中随机选择一个作为负样本
            negative_example_idx = np.random.choice(self.range_all_files, 1)[0]
            negative_example = self.list_all_files[negative_example_idx]
        # anchor，positive，negative
        return positive_example_1, negative_example, positive_example_2


def read_and_resize(filepath):
    im = Image.open((filepath)).convert('RGB')
    im = im.resize((image_size, image_size))
    return np.array(im, dtype="float32")


def augment(im_array):
    if np.random.uniform(0, 1) > 0.9:
        # np.fliplr() 实现矩阵左右翻转
        im_array = np.fliplr(im_array)
    return im_array


def gen(triplet_gen):
    while True:
        list_positive_examples_1 = []
        list_negative_examples = []
        list_positive_examples_2 = []

        for i in range(batch_size):
            positive_example_1, negative_example, positive_example_2 = triplet_gen.get_sample()
            path_pos1 = join(path_train, positive_example_1)
            path_neg = join(path_train, negative_example)
            path_pos2 = join(path_train, positive_example_2)

            positive_example_1_img = read_and_resize(path_pos1)
            negative_example_img = read_and_resize(path_neg)
            positive_example_2_img = read_and_resize(path_pos2)

            positive_example_1_img = augment(positive_example_1_img)
            negative_example_img = augment(negative_example_img)
            positive_example_2_img = augment(positive_example_2_img)

            list_positive_examples_1.append(positive_example_1_img)
            list_negative_examples.append(negative_example_img)
            list_positive_examples_2.append(positive_example_2_img)

        A = preprocess_input(np.array(list_positive_examples_1))
        B = preprocess_input(np.array(list_positive_examples_2))
        C = preprocess_input(np.array(list_negative_examples))

        label = None

        yield ({'anchor_input': A, 'positive_input': B, 'negative_input': C}, label)

def triplet_loss(inputs, dist='euclidean', margin='maxplus', margin_val=0.5):
    anchor, positive, negative = inputs
    positive_distance = K.square(anchor - positive)
    negative_distance = K.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = K.sqrt(K.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = K.sqrt(K.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = K.sum(positive_distance, axis=-1, keepdims=True)
        negative_distance = K.sum(negative_distance, axis=-1, keepdims=True)
    loss = positive_distance - negative_distance
    if margin == 'maxplus':
        loss = K.maximum(0.0, margin_val + loss)
    elif margin == 'softplus':
        loss = K.log(margin_val + K.exp(loss))
    return K.mean(loss)

def triplet_loss_np(inputs, dist='sqeuclidean', margin='maxplus'):
    anchor, positive, negative = inputs
    positive_distance = np.square(anchor - positive)
    negative_distance = np.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = np.sqrt(np.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = np.sqrt(np.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = np.sum(positive_distance, axis=-1, keepdims=True)
        negative_distance = np.sum(negative_distance, axis=-1, keepdims=True)
    loss = positive_distance - negative_distance
    if margin == 'maxplus':
        loss = np.maximum(0.0, 1 + loss)
    elif margin == 'softplus':
        loss = np.log(1 + np.exp(loss))
    return np.mean(loss)


def GetModel():
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='max')
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Dropout(0.6)(x)
    x = Dense(embedding_dim)(x)
    x = Lambda(lambda x: K.l2_normalize(x, axis=1))(x)
    embedding_model = Model(base_model.input, x, name="embedding")

    input_shape = (image_size, image_size, 3)
    anchor_input = Input(input_shape, name='anchor_input')
    positive_input = Input(input_shape, name='positive_input')
    negative_input = Input(input_shape, name='negative_input')
    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)

    inputs = [anchor_input, positive_input, negative_input]
    outputs = [anchor_embedding, positive_embedding, negative_embedding]

    triplet_model = Model(inputs, outputs)
    triplet_model.add_loss(K.mean(triplet_loss(outputs)))

    return embedding_model, triplet_model

data = pd.read_csv(path_csv)
train, test = train_test_split(data, train_size=0.8, random_state=1337)
file_id_mapping_train = {k: v for k, v in zip(train.Image.values, train.Id.values)}
file_id_mapping_test = {k: v for k, v in zip(test.Image.values, test.Id.values)}
gen_tr = gen(sample_gen(file_id_mapping_train))
gen_te = gen(sample_gen(file_id_mapping_test))

checkpoint = ModelCheckpoint(filepath="./data/model_playground/checkpoint_triplet-{epoch:05d}-{loss:.3f}.h5",
                             monitor='val_loss', verbose=2, save_best_only=True, mode='min', period=1)
early = EarlyStopping(monitor="val_loss", mode="min", patience=5)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', factor=0.1, min_lr=1e-6)
callbacks_list = [checkpoint, reduce_lr]

embedding_model, triplet_model = GetModel()

for layer in embedding_model.layers[178:]:
    layer.trainable = True
for layer in embedding_model.layers[:178]:
    layer.trainable = False

triplet_model.compile(loss=None, optimizer=Adam(0.005))
history = triplet_model.fit_generator(gen_tr,
                              validation_data=gen_te,
                              epochs=1000,
                              verbose=2,
                              workers=4,
                              steps_per_epoch=200,
                              validation_steps=50,use_multiprocessing=True,
                              callbacks=callbacks_list)

# nohup python3 -u  triplet_loss.py   > playground_logs.out 2>&1 &