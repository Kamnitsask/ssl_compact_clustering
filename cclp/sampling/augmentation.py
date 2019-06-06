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

import numpy as np
import scipy.stats as st
from math import ceil

import tensorflow as tf

# http://tflearn.org/data_augmentation/
# https://github.com/tflearn/tflearn/blob/master/tflearn/data_augmentation.py
# tf.image.random_flip_left_right / _random_brightness / _contrast / .per_image_whitening

def sample_binary_mask_for_batch(batch_size, p):
    randu = tf.random_uniform( shape=[batch_size], minval=0., maxval=1., dtype=tf.float32)
    randu = tf.cast(tf.less(randu, p), tf.float32)
    randu = tf.expand_dims(randu, axis=1)
    randu = tf.expand_dims(randu, axis=1)
    randu = tf.expand_dims(randu, axis=1) # make the 1d vector randu to 2d/3d/4d sequentially, adding x,y,c dimensions.
    return randu
    
def sample_multiscale_mask_for_batch(batch_size, num_scales):
    p_per_scale = 1./num_scales
    randu = tf.random_uniform( shape=[batch_size], minval=0., maxval=1., dtype=tf.float32)
    rand_mask = tf.zeros(shape=[batch_size])
    for scale_i in range(num_scales):
        bool_mask = tf.logical_and( tf.greater_equal(randu, p_per_scale*scale_i) , tf.less(randu, p_per_scale*(scale_i+1)) )
        rand_mask += tf.cast(bool_mask, tf.float32) * scale_i
    rand_mask = tf.expand_dims(rand_mask, axis=1)
    rand_mask = tf.expand_dims(rand_mask, axis=1)
    rand_mask = tf.expand_dims(rand_mask, axis=1) # make the 1d vector randu to 2d/3d/4d sequentially, adding x,y,c dimensions.
    return rand_mask

def random_color_invert(inp_batch, params, db_zero_centered):
    # Apply augmentation. This is just randomly (50%) inverting the image's colors by abs(image-255).
    # inp_batch.shape = [n, x, y, chans]
    randu = sample_binary_mask_for_batch( int(inp_batch.shape[0]), params['p'] ) # p is the probability to augment

    if db_zero_centered:
        LOG.debug("=================== ZERO CENTERED = TRUE ===================")
        return (1-randu)*inp_batch - randu*inp_batch
    else:
        LOG.debug("=================== ZERO CENTERED = FALSE ===================")
        inp_batch = tf.cast(inp_batch, tf.float32)
        max_int = tf.reduce_max(input_tensor=inp_batch, axis=(0,1,2), keep_dims=True)
        min_int = tf.reduce_min(input_tensor=inp_batch, axis=(0,1,2), keep_dims=True)
        return inp_batch*(1-randu) + ( max_int + min_int - inp_batch )*randu
    # return tf.abs(inp_batch - tf.reduce_max(input_tensor=inp_batch, axis=None, keep_dims=True) * randu)
    
def random_affine_transf( inp_batch, params ):
    # Is this their original aff transf routine? : https://github.com/csgaobb/learning_by_association/blob/master/semisup/train.py
    
    # inp_batch.shape = [n, x, y, chans]
    
    transf = params['transf']
    # First, pad each dimension with as much as the affine transf wanted.
    # for each dimension of the batch (rows), padding before and after (2 cols)
    num_dims_xyz = len(inp_batch.shape)-2
    paddings_per_dim = [[0]*num_dims_xyz,
                        [transf[0]]*num_dims_xyz,
                        [transf[1]]*num_dims_xyz,
                        [0]*num_dims_xyz ]
    padded_batch = tf.pad( tensor=inp_batch,
                           paddings=paddings_per_dim,
                           mode='SYMMETRIC', # or SYMMETRIC, REFLECT, or CONSTANT
                           #constant_values=0
                        ) 
    augm_batch = tf.random_crop( value=padded_batch, size=inp_batch.shape )
    return augm_batch
    

def get_gaussian_kernel_np(sigma, num_of_dims):
    # Returns a numpy array of shape: [x, y, (z)], that is a kernel of width_of_kernel, with values given by a gaussian with sigma std and dimensionality num_of_dims.
    # First initialise a single 2d/3d filter that looks like a truncated gaussian
    # In the LBA paper, kernel width = 7 with sigma ~1.3 was used.
    width_of_kern = int( ceil(sigma * 4) + 1 ) # Kernel is truncated at 2 stds both ways (95% rule). --- _pdf and cdfcent have +1 here, others had (sigma*4)
    x = np.linspace( -sigma, sigma, width_of_kern )
    kern1d = st.norm.pdf(x)
    kern_multidim = np.outer(kern1d, kern1d) # from 1d make it 2d. Essentially the 2d covariance matrix.
    kern_multidim = np.outer(kern_multidim, kern1d) if num_of_dims == 3 else kern_multidim
    kernel_raw = kern_multidim
    kernel_normed = kernel_raw/kernel_raw.sum() # normalize the pdf
    return kernel_normed

def conv_depthwise_with_given_npkern(kern_np, inp_batch):
    # Make numpy kernel tensor ready for a conv
    initial_tensor = np.array(kern_np, dtype = np.float32)
    initial_tensor = initial_tensor.reshape( list(kern_np.shape) + [1, 1] ) # to add the last 2 unary dims.
    initial_tensor = np.repeat(initial_tensor, inp_batch.shape[-1], axis = 2 ) # repeat for the input chans
    
    with tf.variable_scope('net', reuse=tf.AUTO_REUSE):
        kernel_tens = tf.Variable(initial_tensor, name='blur_kern', dtype=tf.float32, trainable=False)
        augm_batch = tf.nn.depthwise_conv2d(inp_batch, kernel_tens, strides=[1]*len(kernel_tens.shape), padding='SAME')
        
    #tf.summary.image('blur_aug_kern_sc'+str(scale_i),  (np.array(kern_np, dtype = np.float32)).reshape( [1]+list(kern_np.shape)+[1] ) , max_outputs=3, collections=[tf.GraphKeys.SUMMARIES])
    return augm_batch

def random_gauss_blur(inp_batch, params):
    # https://pypkg.com/pypi/tflearn/f/tflearn/data_augmentation.py --> add_random_blur
    # https://github.com/antonilo/TensBlur/blob/master/smoother.py
    sigma = float(params['s']) # max sigma
    num_scales = params['sc']
    num_of_dims = len(inp_batch.shape)-2
    
    # shapeOfFilter for depthwise_conv2d : [ height, width, inChans, outputChanMultiplier ] if in_tensor is "NHWC" (default).
    augm_batches_all_scales = []
    for scale_i in range(num_scales):
        sigma_for_scale = 1.0*sigma/(scale_i+1)
        kernel_normed = get_gaussian_kernel_np(sigma_for_scale, num_of_dims)
        augm_batch_for_scale = conv_depthwise_with_given_npkern(kernel_normed, inp_batch)
        augm_batches_all_scales.append(augm_batch_for_scale)
    
    mask_of_scales = sample_multiscale_mask_for_batch( int(inp_batch.shape[0]), num_scales)
    
    augm_batch = augm_batches_all_scales[0] * tf.cast(tf.equal(mask_of_scales, 0), tf.float32) # will be equal to augm_scale0 for the samples of the batch that scale0 was chosen, and 0 elsewhere.
    for scale_i in range(1, num_scales):
        augm_batch += augm_batches_all_scales[scale_i] * tf.cast(tf.equal(mask_of_scales, scale_i), tf.float32)
        
    randu = sample_binary_mask_for_batch( int(inp_batch.shape[0]), params['p'] ) # 0 means use origin. 1 means augment.
    return inp_batch * (tf.cast(tf.equal(randu,0), tf.float32)) + augm_batch * (tf.cast(tf.equal(randu,1), tf.float32))
    

def random_reflect(inp_batch):
    flip = lambda x: tf.image.random_flip_left_right( x )
    return tf.map_fn(flip, inp_batch)
    


def augment_tens(inp_batch, params, db_zero_centered):
    # inp_batch is a tensor.
    augm_inp = inp_batch
    if params is None:
        return inp_batch
    if "color_inv" in params and params["color_inv"]["apply"]:
        augm_inp = random_color_invert(augm_inp, params["color_inv"]["params"], db_zero_centered)
    if "rand_crop" in params and params["rand_crop"]["apply"]:
        augm_inp = random_affine_transf(augm_inp, params["rand_crop"]["params"])
    if "blur" in params and params["blur"]["apply"]:
        augm_inp = random_gauss_blur(augm_inp, params["blur"]["params"])
    if "reflect" in params and params["reflect"]["apply"]:
        augm_inp = random_reflect(augm_inp)
    return augm_inp
    


