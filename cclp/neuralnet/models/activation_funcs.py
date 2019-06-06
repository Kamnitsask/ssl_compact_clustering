#!/usr/bin/env python

# Copyright (c) 2018, Konstantinos Kamnitsas
#
# This program is free software; you can redistribute and/or modify
# it under the terms of the Apache License, Version 2.0. See the 
# accompanying LICENSE file or read the terms at:
# http://www.apache.org/licenses/LICENSE-2.0

from __future__ import absolute_import, division, print_function

import logging
LOG = logging.getLogger('main')

import tensorflow as tf

def linear(x, name):
    return x

def lrelu(x, alpha=0.1, name=None):
  return tf.identity( tf.nn.relu(x) - alpha * tf.nn.relu(-x), name=name )

def get_act_func(act_str, name=None):
    # if i need to pass paramas to act funcs, pass params here, and return partials.
    if act_str == "relu":
        return tf.nn.relu
    elif act_str == "elu":
        return tf.nn.elu
    elif act_str == "lrelu":
        return lrelu
    elif act_str == "tanh":
        return tf.nn.tanh
    elif act_str == "linear" or act_str == "lin" or act_str == "line":
        return linear
    elif act_str is None:
        return None
    
    raise NotImplemented










