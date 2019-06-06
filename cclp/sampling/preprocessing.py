#!/usr/bin/env python

# Copyright (c) 2018, Konstantinos Kamnitsas
#
# This program is free software; you can redistribute and/or modify
# it under the terms of the Apache License, Version 2.0. See the 
# accompanying LICENSE file or read the terms at:
# http://www.apache.org/licenses/LICENSE-2.0

from __future__ import division, print_function

import logging
LOG = logging.getLogger('main')

import tensorflow as tf

# Important resources:
# https://www.tensorflow.org/api_guides/python/image


def z_score_per_case(tens_batch):
    # per channel per sample
    mean = tf.reduce_mean(tens_batch, axis=[1, 2], keep_dims=True)
    std = tf.reduce_mean(tf.square(tens_batch - mean), axis=[1, 2], keep_dims=True)
    tens_batch = (tens_batch - mean) / (std + 1e-5)
    return tens_batch

# similar to the one in data_utils.img_processing.normalize_imgs(), but that one operates on np arrays, before augmentation.
def normalize_tens(tens_batch, norm_type):
    norm_tens = None
    if norm_type is None:
        norm_tens = tens_batch
    elif norm_type == "zscorePerCase":
        norm_tens = z_score_per_case(tens_batch)
    
    return norm_tens









