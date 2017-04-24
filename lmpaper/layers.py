# -*- coding: utf-8 -*-
# Project : LM4paper
# Created by igor on 2016/12/13

import tensorlayer as tl
import tensorflow as tf


class TowDEmbeddingInputLayer(tl.layers.Layer):
    """
    The :class:`EmbeddingInputlayer` class is a fully connected layer,
    for Word Embedding. Words are input as integer index(row and column).
    The output is the embedded word vector.
    Every word has two vector: row vector and column vector


    :param layer:
    :param name:
    """

    def __init__(self, layer=None, name="2d_embedding_layers"):

        pass

        tl.utils.fit()