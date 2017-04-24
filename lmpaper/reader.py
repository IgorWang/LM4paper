#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author igor
# Created by iFantastic on 16-5-12
'''
数据读取
根据句子的长短利用fixlength reader 读取数据
'''

# -*- coding: utf-8 -*-
#
#
# Author: Igor
import os

import tensorflow as tf

from lmpaper.context import DATA_DIR, TrainConfig

SEQUENCE_LENGTH = 50
SLICE_LENGTH = TrainConfig().num_steps + 1

assert SLICE_LENGTH < SEQUENCE_LENGTH, "wrong length"

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 9600000 // 10
TOTAL_EXAMLES = 9600000


def read_data(filename_queue):
    '''
    数据的读取
    :param filename_queue:
    :return:
    '''

    class DataRecord(object):
        pass

    result = DataRecord()

    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    record_defaults = [[1] for i in range((SEQUENCE_LENGTH))]
    record = tf.decode_csv(value,
                           record_defaults=record_defaults)
    # 特征
    result.feature = tf.pack(record[0:SLICE_LENGTH - 1])
    # 标签
    result.label = tf.pack(record[1:SLICE_LENGTH])

    return result


def read_data_with_bukcets(filename_queue, sequence_length):
    '''
    数据的读取
    :param filename_queue:
    :return:
    '''

    class DataRecord(object):
        pass

    result = DataRecord()

    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    record_defaults = [[1] for i in range((sequence_length))]
    record = tf.decode_csv(value,
                           record_defaults=record_defaults)
    # 特征
    result.feature = tf.pack(record[0:sequence_length - 1])
    # 标签
    result.label = tf.pack(record[1:sequence_length])

    return result


def _generate_features_and_labels_batch(feature, label, min_queue_examples, batch_size, shuffle):
    '''
    构建特征和标签的批量队列
    :param features: 1-D Tensor of [SEQUENCE_LENGTH]
    :param lables: 1-D Tensor of [SEQUENCE_LENGTH]
    :param min_queue_examples:
    :param batch_size:批量的大小
    :param shuffle:
    :return:
        features:2-D [batch_size,SEQUENCE_LENGTH]
        labels:2-D [batch_size,SEQUENCE_LENGTH]
    '''
    num_preprocess_threads = 4
    if shuffle:
        features, label_batch = tf.train.shuffle_batch(
            [feature, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        features, label_batch = tf.train.batch(
            [feature, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    return features, label_batch


def inputs_with_buckets(file_with_buckets, buckets, batch_size):
    '''
    :param file_with_buckets:每个bucket对应的文件
    :param buckets: 不同大小的buckets列表,示例-[15, 25, 35, 50]
    根据buckets获取语料
    :return: list of Tensor示例-[Tensor(batch_size,15),
            Tensor(batch_size,25),Tensor(batch_size,35),Tensor(batch_size,50)]
    '''
    inputs_wb = []
    targets_wb = []
    for i, sl in enumerate(buckets):
        filename_queue = tf.train.string_input_producer([file_with_buckets[i]])

        read_input = read_data_with_bukcets(filename_queue, sl)

        min_fraction_of_examples_in_queue = 0.001
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                 min_fraction_of_examples_in_queue)

        print('Filling queue with %d case before starting to text.'
              'This will take a few minutes.' % min_queue_examples)

        x, y = _generate_features_and_labels_batch(read_input.feature, read_input.label,
                                                   min_queue_examples, batch_size,
                                                   shuffle=True)
        inputs_wb.append(x)
        targets_wb.append(y)
    return inputs_wb, targets_wb


def distorted_inputs(filenames, batch_size):
    '''
    构建文本输入 - 训练
    :param queue: list of file name
    :param batch_size: 批量大小
    :return:
        document: 2D tensor of [batch_size,SEQUENCE_LENGTH]
        labels: 2D tensor [batch_size,SEQUENCE_LENGTH]
    '''
    # for f in filenames:
    #     if not os.path.exists(f):
    #         raise ValueError("Failed to find file:" + f)

    # 文件的读取队列
    filename_queue = tf.train.string_input_producer(filenames)

    read_input = read_data(filename_queue)

    min_fraction_of_examples_in_queue = 0.004
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)

    print('Filling queue with %d case before starting to text.'
          'This will take a few minutes.' % min_queue_examples)

    return _generate_features_and_labels_batch(read_input.feature, read_input.label,
                                               min_queue_examples, batch_size,
                                               shuffle=True)


if __name__ == '__main__':
    train_file_path = [os.path.join(DATA_DIR, 'train', 'train.csv')]
    input_x, input_y = distorted_inputs(train_file_path, 2)
    with tf.Session() as sess:
        threads = tf.train.start_queue_runners()
        for i in range(3):
            x, y = sess.run([input_x, input_y])
            print(x)
