#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author igor
# Created by iFantastic on 16-4-25
'''

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units, use as embedding size
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
'''

import os
import tensorflow as tf

# 项目路径
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
# 数据路径
DATA_DIR = os.path.join(BASE_DIR, 'data')

BUCKETS = [15, 25, 35, 50]
FLAGS = tf.app.flags.FLAGS
flags = tf.flags
flags.DEFINE_string('raw_dir', os.path.join(DATA_DIR, 'raw'), 'raw data file directory')
flags.DEFINE_string('model_dir', os.path.join(DATA_DIR, 'model'), 'model directroy')



class TrainConfig(object):
    '''
    模型的参数
    '''
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 1
    num_steps = 30
    hidden_size = 200
    max_epoch = 10
    max_max_epoch = 100
    keep_prob = 0.8
    lr_decay = 0.9
    batch_size = 10
    vocab_size = 235809


class TestConfig(object):
    '''
    预测时的参数
    '''
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 1
    num_steps = 30
    hidden_size = 200
    max_epoch = 1
    max_max_epoch = 2
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 1
    vocab_size = 235809
