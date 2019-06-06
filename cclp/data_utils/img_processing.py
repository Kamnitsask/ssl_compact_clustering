#!/usr/bin/env python

# Copyright (c) 2018, Konstantinos Kamnitsas
#
# This program is free software; you can redistribute and/or modify
# it under the terms of the Apache License, Version 2.0. See the 
# accompanying LICENSE file or read the terms at:
# http://www.apache.org/licenses/LICENSE-2.0

from __future__ import division
from __future__ import print_function

import logging
LOG = logging.getLogger('main')

import numpy as np
from scipy import linalg

def z_score_per_case(imgs):
    # per channel per sample
    mean = np.mean(imgs, axis=(1, 2), keepdims=True, dtype="float32")
    std = np.std(imgs, axis=(1, 2), keepdims=True, dtype="float32")
    imgs = (imgs - mean) / (std + 1e-5)
    return imgs
    
def z_score_db(imgs, stats_from_train_to_use_in_test=None):
    # Z score norm with stats across database
    if stats_from_train_to_use_in_test is None:
        #mean_int = np.mean(imgs, dtype="float32")
        #stddev_int = np.std(imgs, dtype="float32")
        mean_int = np.mean(imgs, axis=(0,1,2), keepdims=True, dtype="float32")
        stddev_int = np.std(imgs, axis=(0,1,2), keepdims=True, dtype="float32")
    else : # stats have already been given, normalize with them.
        mean_int = stats_from_train_to_use_in_test["mean"]
        stddev_int = stats_from_train_to_use_in_test["std"]
        
    imgs = (imgs - mean_int) / stddev_int
    return imgs

def centre0sc1(imgs, stats_from_train_to_use_in_test=None):
    # Centers and rescales the database.
    if stats_from_train_to_use_in_test is None:
        min_int = np.min(imgs, axis=(0,1,2), keepdims=True).astype("float32")
        max_int = np.max(imgs, axis=(0,1,2), keepdims=True).astype("float32")
    else:
        min_int = stats_from_train_to_use_in_test["min"]
        max_int = stats_from_train_to_use_in_test["max"]
        
    half_diff_max_min = (max_int - min_int) / 2.
    middle_int = min_int + half_diff_max_min
    imgs = (imgs - middle_int) / half_diff_max_min
    return imgs

def rescale01(imgs, stats_from_train_to_use_in_test=None):
    # Rescales the db, setting min int to 0, and max int to 1.
    if stats_from_train_to_use_in_test is None:
        min_int = np.min(imgs, axis=(0,1,2), keepdims=True).astype("float32")
        max_int = np.max(imgs, axis=(0,1,2), keepdims=True).astype("float32")
    else:
        min_int = stats_from_train_to_use_in_test["min"]
        max_int = stats_from_train_to_use_in_test["max"]
        
    imgs = (imgs - min_int) / (max_int - min_int)
    return imgs

# From temporal ensembles: https://github.com/smlaine2/tempens/blob/master/zca_bn.py
class ZCA(object):
    def __init__(self, regularization=1e-5, x=None):
        self.regularization = regularization
        if x is not None:
            self.fit(x)

    def fit(self, x):
        s = x.shape
        x = x.copy().reshape((s[0],np.prod(s[1:]))).astype("float32")
        m = np.mean(x, axis=0).astype("float32")
        x -= m
        sigma = np.dot(x.T,x) / x.shape[0]
        U, S, V = linalg.svd(sigma)
        tmp = np.dot(U, np.diag(1./np.sqrt(S+self.regularization)))
        tmp2 = np.dot(U, np.diag(np.sqrt(S+self.regularization)))
        self.ZCA_mat = np.dot(tmp, U.T).astype("float32")
        self.inv_ZCA_mat = np.dot(tmp2, U.T).astype("float32")
        self.mean = m

    def apply(self, x):
        s = x.shape
        return np.dot(x.reshape((s[0],np.prod(s[1:]))) - self.mean, self.ZCA_mat).reshape(s)
            
    def invert(self, x):
        s = x.shape
        return (np.dot(x.reshape((s[0],np.prod(s[1:]))), self.inv_ZCA_mat) + self.mean).reshape(s)
    
def normalize_imgs(imgs, norm_type, stats_from_train_to_use_in_test=None):
    
    norm_imgs = None
    zca_instance = None
    
    if norm_type is None:
        norm_imgs = imgs
    elif norm_type == "zscoreDb":
        norm_imgs = z_score_db(imgs)
    elif norm_type == "zscorePerCase":
        norm_imgs = z_score_per_case(imgs)
    elif norm_type == "center0sc1":
        norm_imgs = centre0sc1(imgs)
    elif norm_type == "rescale01":
        norm_imgs = rescale01(imgs)
    elif norm_type == "zca" or norm_type == "zca_center0sc1":
        if stats_from_train_to_use_in_test is None or stats_from_train_to_use_in_test["zca_instance"] is None: # Training time
            LOG.debug("ZCA Whitening. Creating a new instance.")
            zca_instance = ZCA(x=imgs)
        else: # Testing time
            LOG.debug("ZCA Whitening. Using given instance. Testing time?!")
            zca_instance = stats_from_train_to_use_in_test["zca_instance"]
        norm_imgs = zca_instance.apply(imgs)
        
        if norm_type == "zca_center0sc1": # Now also apply zscoreDb
            LOG.debug("ZCA Whitening, also applying "+str(norm_type)+" now! mean="+str(np.mean(norm_imgs))+" std="+str(np.std(norm_imgs)) )
            
            norm_imgs = centre0sc1(norm_imgs)
    
    return norm_imgs, zca_instance




