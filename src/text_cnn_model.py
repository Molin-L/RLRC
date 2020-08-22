#!/usr/bin/env python
# -*- coding: utf-8 -*-
# by vellhe 2018/4/7
import math

from keras import Sequential, Input, Model
from keras.layers import Flatten, Dense, Dropout, Conv2D, Concatenate, \
    MaxPool2D
from keras.regularizers import l2

from base_model.base_model_creator import BaseModelCreator


class TextCNN(BaseModelCreator):
    '''
    TextCNN模型实现
    '''

    def __init__(self, config):
        self.config = dict()
        self.config["word_num"] = 50
        self.config["embedding_dim"] = 100
        self.config["channels"] = 1
        self.config["labels_num"] = 2

        self.config["filter_sizes"] = [3, 4, 5]
        self.config["pool_sizes"] = [48, 47, 46]  # default value: word_num - filter_size + 1
        self.config["num_filters"] = 128
        self.config["dropout"] = 0.5
        self.config["cnn_activation"] = "elu"
        self.config["dense_units"] = 1024
        self.config["dense_activation"] = "elu"
        self.config["weight_decay"] = 0

        self.config.update(config)
        super().__init__()

    def create(self, weights_path=None):
        '''
        channel_last (rows, cols, channels) <=> (word_num, embedding_dim, channels)
        :param weights_path:
        :return:
        '''
        word_num = self.config["word_num"]
        embedding_dim = self.config["embedding_dim"]
        channels = self.config["channels"]
        labels_num = self.config["labels_num"]

        filter_sizes = self.config["filter_sizes"]
        pool_sizes = self.config["pool_sizes"]
        num_filters = self.config["num_filters"]
        dropout = self.config["dropout"]
        dense_units = self.config["dense_units"]
        cnn_activation = self.config["cnn_activation"]
        dense_activation = self.config["dense_activation"]
        weight_decay = self.config["weight_decay"]

        # 文本内容卷积
        input_tensor = Input((word_num, embedding_dim, channels))

        conv_blocks = list()
        for idx, filter_size in enumerate(filter_sizes):
            conv = Conv2D(num_filters, kernel_size=(filter_size, embedding_dim), padding='valid',
                          kernel_initializer='normal',
                          kernel_regularizer=l2(weight_decay),
                          # activity_regularizer=regularizers.l1(weight_decay),
                          activation=cnn_activation)(input_tensor)

            maxpool = MaxPool2D(pool_size=(math.ceil(pool_sizes[idx]), 1),
                                strides=(int(pool_sizes[idx]), 1), padding='valid')(conv)
            conv_blocks.append(maxpool)

        concatenated_tensor = Concatenate(axis=1)(conv_blocks)
        flatten = Flatten()(concatenated_tensor)
        output_tensor = Dropout(dropout)(flatten)

        model_content = Model(inputs=input_tensor, outputs=output_tensor)

        # 分类
        model = Sequential()
        model.add(model_content)
        model.add(Dense(dense_units, activation=dense_activation))
        model.add(Dropout(dropout))
        model.add(Dense(labels_num, activation="softmax"))

        model_content.summary(positions=[.33, .60, .77, 1.])
        model.summary(positions=[.33, .60, .77, 1.])

        if weights_path:
            model.load_weights(weights_path)

        self.keras_model = model
        return self.keras_model
