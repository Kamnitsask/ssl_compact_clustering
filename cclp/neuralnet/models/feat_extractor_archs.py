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

from cclp.neuralnet.models.activation_funcs import get_act_func

# Architectures to use for a feature extractor z.

def cifar_cnn( inputs,
               is_training,
               emb_z_size=128,
               emb_z_act="elu",
               l2_weight=1e-5,
               batch_norm_decay=0.99,
               seed=None):
    # Ala iGAN & badGAN. TriGan used bigger.
    
    fwd = inputs
    tf.summary.scalar('min_int_net_in', tf.reduce_min(fwd), collections=[tf.GraphKeys.SUMMARIES] if is_training else ["eval_summaries"])
    tf.summary.scalar('max_int_net_in', tf.reduce_max(fwd), collections=[tf.GraphKeys.SUMMARIES] if is_training else ["eval_summaries"])
    
    # Helper about defaults: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py
    
    weights_initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=seed, dtype=tf.float32)
    #weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, seed=seed, dtype=tf.float32) # He et al.
    
    weights_regularizer = tf.contrib.layers.l2_regularizer
    
    activation_fn = get_act_func(emb_z_act)
    
    USE_BN = batch_norm_decay != -1 # True if not -1. BN puts in tf.GraphKeys.UPDATE_OPS, needs to be put in train_op.
    BN_TRAINABLE = True
    
    max_pool = tf.contrib.layers.max_pool2d
    avg_pool = tf.contrib.layers.avg_pool2d
    dropout_f = tf.layers.dropout # tf.layers.dropout gets drop_rate. Others get keep_rate.
    
    if is_training:
        fwd = fwd + tf.random_normal( tf.shape(fwd), mean=0.0, stddev=0.03, seed=seed, dtype=tf.float32 )
    
    fwd = tf.contrib.layers.conv2d(fwd, 96, [3, 3], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                                   weights_regularizer=weights_regularizer(l2_weight), scope='c1_1')
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn1_1" ) if USE_BN else fwd # BN puts in tf.GraphKeys.UPDATE_OPS, needs to be put in train_op.
    fwd = activation_fn(fwd)
    fwd = tf.contrib.layers.conv2d(fwd, 96, [3, 3], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c1_2')
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn1_2" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    fwd = tf.contrib.layers.conv2d(fwd, 96, [3, 3], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c1_3')
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn1_3" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    fwd = max_pool(fwd, [2, 2], scope='p1')  # 14
    
    fwd = dropout_f( fwd, rate=0.5, training=is_training, seed=seed )
    fwd = tf.contrib.layers.conv2d(fwd, 192, [3, 3], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c2_1')
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn2_1" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    fwd = tf.contrib.layers.conv2d(fwd, 192, [3, 3], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c2_2')
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn_2_2" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    fwd = tf.contrib.layers.conv2d(fwd, 192, [3, 3], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c2_3')
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn2_3" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    fwd = max_pool(fwd, [2, 2], scope='p2')  # 7
    
    fwd = dropout_f( fwd, rate=0.5, training=is_training, seed=seed )
    fwd = tf.contrib.layers.conv2d(fwd, 192, [3, 3], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c3_1')
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn3_1" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    fwd = tf.contrib.layers.conv2d(fwd, 192, [3, 3], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c3_2')
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn3_2" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    # Dropout should NOT be used right before the graph construction (emb_z). It makes the affinity matrix too sparse. Rather, at least a conv before, such as here for example.
    # fwd = dropout_f( fwd, rate=0.5, training=is_training, seed=seed ) # Doesnt do much, but to show there is no problem training with dropout here.
    fwd = tf.contrib.layers.conv2d(fwd, emb_z_size, [3, 3], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c3_3')
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn3_3" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    
    fwd = avg_pool(fwd, fwd.get_shape().as_list()[1:3], scope='p3')
    
    fwd = tf.layers.flatten(fwd)
    
    emb_z = fwd
    return emb_z


def svhn_cnn( inputs,
              is_training,
              emb_z_size=128,
              emb_z_act="elu",
              l2_weight=1e-5,
              batch_norm_decay=0.99,
              seed=None):
    # Ala iGAN & badGAN. TriGan used bigger.
    fwd = inputs    
    tf.summary.scalar('min_int_net_in', tf.reduce_min(fwd), collections=[tf.GraphKeys.SUMMARIES] if is_training else ["eval_summaries"])
    tf.summary.scalar('max_int_net_in', tf.reduce_max(fwd), collections=[tf.GraphKeys.SUMMARIES] if is_training else ["eval_summaries"])
    
    weights_initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=seed, dtype=tf.float32)
    weights_regularizer = tf.contrib.layers.l2_regularizer
    
    activation_fn = get_act_func(emb_z_act)
    
    USE_BN = batch_norm_decay != -1 # True if not -1. BN puts in tf.GraphKeys.UPDATE_OPS, needs to be put in train_op.
    BN_TRAINABLE = True
    
    max_pool = tf.contrib.layers.max_pool2d
    avg_pool = tf.contrib.layers.avg_pool2d
    dropout_f = tf.layers.dropout # tf.layers.dropout gets drop_rate. Others get keep_rate.
    
    if is_training:
        fwd = fwd + tf.random_normal( tf.shape(fwd), mean=0.0, stddev=0.05, seed=seed, dtype=tf.float32 ) # iGan, badGan, triGan use this and/or dropout. Still CCLP great without this.
    fwd = dropout_f( fwd, rate=0.1, training=is_training, noise_shape=(fwd.get_shape().as_list()[0], 1, 1, fwd.get_shape().as_list()[-1]), seed=seed ) # ala badGAN
    
    fwd = tf.contrib.layers.conv2d(fwd, 64, [3, 3], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                                   weights_regularizer=weights_regularizer(l2_weight), scope='c1_1')
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn1_1" ) if USE_BN else fwd # BN puts in tf.GraphKeys.UPDATE_OPS, needs to be put in train_op.
    fwd = activation_fn(fwd)
    fwd = tf.contrib.layers.conv2d(fwd, 64, [3, 3], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c1_2')
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn1_2" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    fwd = tf.contrib.layers.conv2d(fwd, 64, [3, 3], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c1_3')
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn1_3" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    fwd = max_pool(fwd, [2, 2], scope='p1')  # 14
    
    fwd = dropout_f( fwd, rate=0.5, training=is_training, seed=seed )
    fwd = tf.contrib.layers.conv2d(fwd, 128, [3, 3], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c2_1')
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn2_1" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    fwd = tf.contrib.layers.conv2d(fwd, 128, [3, 3], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c2_2')
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn_2_2" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    fwd = tf.contrib.layers.conv2d(fwd, 128, [3, 3], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c2_3')
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn2_3" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    fwd = max_pool(fwd, [2, 2], scope='p2')  # 7
    
    fwd = dropout_f( fwd, rate=0.5, training=is_training, seed=seed )
    fwd = tf.contrib.layers.conv2d(fwd, 128, [3, 3], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c3_1')
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn3_1" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    fwd = tf.contrib.layers.conv2d(fwd, 128, [3, 3], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c3_2')
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn3_2" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    # Dropout should NOT be used right before the graph construction (emb_z). It makes the affinity matrix too sparse. Rather, at least a conv before, such as here for example.
    # fwd = dropout_f( fwd, rate=0.5, training=is_training, seed=seed ) # Doesnt do much, but to show there is no problem training with dropout here.
    fwd = tf.contrib.layers.conv2d(fwd, emb_z_size, [3, 3], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c3_3')
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn3_3" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    
    fwd = avg_pool(fwd, fwd.get_shape().as_list()[1:3], scope='p3')
    
    fwd = tf.layers.flatten(fwd)
    
    emb_z = fwd
    return emb_z



def mnist_cnn( inputs,
               is_training,
               emb_z_size=128,
               emb_z_act="elu",
               l2_weight=1e-5,
               batch_norm_decay=0.99,
               seed=None):
    # inputs: [Batchsize x H x W x Channels]
    # Architecture ala Triple GAN.
    fwd = inputs
    tf.summary.scalar('min_int_net_in', tf.reduce_min(fwd), collections=[tf.GraphKeys.SUMMARIES] if is_training else ["eval_summaries"])
    tf.summary.scalar('max_int_net_in', tf.reduce_max(fwd), collections=[tf.GraphKeys.SUMMARIES] if is_training else ["eval_summaries"])
    
    # weights_initializer = tf.truncated_normal_initializer(stddev=0.01, seed=seed)
    # weights_initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=seed, dtype=tf.float32)
    weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=True, seed=seed, dtype=tf.float32)
    weights_regularizer = tf.contrib.layers.l2_regularizer
    
    activation_fn = get_act_func(emb_z_act)
    
    USE_BN = batch_norm_decay != -1 # True if not -1. BN puts in tf.GraphKeys.UPDATE_OPS, needs to be put in train_op.
    BN_TRAINABLE = True
    
    max_pool = tf.contrib.layers.max_pool2d
    avg_pool = tf.contrib.layers.avg_pool2d
    dropout_f = tf.layers.dropout # tf.layers.dropout gets drop_rate. Others get keep_rate.
    
    if is_training:
        fwd = fwd + tf.random_normal( tf.shape(fwd), mean=0.0, stddev=0.3, seed=seed, dtype=tf.float32 ) # iGan, badGan, triGan use this and/or dropout
        
    fwd = tf.contrib.layers.conv2d(fwd, 32, [3, 3], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                                   weights_regularizer=weights_regularizer(l2_weight), scope='c1_1')
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn1_1" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    fwd = tf.contrib.layers.conv2d(fwd, 32, [3, 3], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c1_2')
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn1_2" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    fwd = max_pool(fwd, [2, 2], scope='p1')  # 14
    
    fwd = tf.contrib.layers.conv2d(fwd, 64, [3, 3], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c2_1')
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn2_1" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    fwd = tf.contrib.layers.conv2d(fwd, 64, [3, 3], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c2_2')
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn_2_2" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    fwd = max_pool(fwd, [2, 2], scope='p2')  # 7
    
    fwd = tf.contrib.layers.conv2d(fwd, 128, [3, 3], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c3_1')
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn3_1" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    # Dropout should NOT be used right before the graph construction (emb_z). It makes the affinity matrix too sparse. Rather, at least a conv before, such as here for example.
    fwd = dropout_f( fwd, rate=0.1, training=is_training, noise_shape=(fwd.get_shape().as_list()[0], 1, 1, fwd.get_shape().as_list()[-1]), seed=seed ) # ala badGan. Doesnt do much, but to show there is no problem training with dropout here.
    fwd = tf.contrib.layers.conv2d(fwd, emb_z_size, [3, 3], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c3_2')
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn3_2" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    
    fwd = avg_pool(fwd, fwd.get_shape().as_list()[1:3], scope='p3')
    
    fwd = tf.layers.flatten(fwd)
    
    emb_z = fwd
    return emb_z



def get_feat_extractor_z_func( feat_extractor_z ):
    # Returns: Pointer to the function for the architecture of feature extractor Z.
    arch_func = None
    if feat_extractor_z == "mnist_cnn":
        arch_func = mnist_cnn
    elif feat_extractor_z == "svhn_cnn":
        arch_func = svhn_cnn
    elif feat_extractor_z == "cifar_cnn":
        arch_func = cifar_cnn
    return arch_func







