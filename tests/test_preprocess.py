# -*- coding: utf-8 -*-
# Project : LM4paper
# Created by igor on 2016/12/12

import pytest
import os
from utils import DatasetReader

__author__ = "Igor"
__copyright__ = "Igor"
__license__ = "gpl3"


def test_DatasetReader():
    dir = "/data/temp/"
    pattern = ".*"

    filenames = [os.path.join(dir, i) for i in os.listdir(dir)]

    for fn in DatasetReader(dir, pattern):
        assert fn in filenames
