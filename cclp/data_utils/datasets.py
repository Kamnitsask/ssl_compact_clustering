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

from cclp.data_utils.img_processing import normalize_imgs

class DataSet(object):
    def __init__(   self,
                    images,
                    labels,
                    reshape=False):
        # images: np.array: [numberOfSamples, x, y, z, channels ]
        # labels: np.array: [numberOfSamples] or [numberOfSamples, numOfClasses] if it's one hot.
        # reshape: boolean, specifying whether to flatten each image in one vector (True) or not. Eg for MLP.
        assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]
        self._originalShapeOfAnImg = images.shape[1:]
        self._num_classes = int(np.max(labels) + 1)
        
        # Convert shape from [num examples, rows, columns, channels]
        # to [num examples, rows*columns*channels]
        if reshape:
            images = images.reshape( (images.shape[0], -1) )
        self._samples = images # called samples, because they have been reshaped. they are no longer images.
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        
        # For the normalization:
        self._update_stats(norm_type=None, zca_instance=None) # Used by normalize.
    
    
    def _update_stats(self, norm_type, zca_instance):
        norm_type_is_zca = norm_type is not None and norm_type.find("zca") > -1
        self._db_stats = {  "norm_type": norm_type,
                            "db_zero_centered": ( norm_type in ["zscoreDb", "zscorePerCase"] ), # zscoreDb worked fine with both inverse-augms.
                            "min": np.min(self._samples, axis=(0,1,2), keepdims=True)*1.,
                            "max": np.max(self._samples, axis=(0,1,2), keepdims=True)*1.,
                            "mean": np.mean(self._samples, axis=(0,1,2), keepdims=True, dtype="float32"),
                            "std": np.std(self._samples, axis=(0,1,2), keepdims=True, dtype="float32"),
                            "zca_instance": zca_instance if norm_type_is_zca else None }
        
    def getShapeOfAnImage(self):
        return self._originalShapeOfAnImg
    def getShapeOfASample(self):
        return self._samples.shape[1:]
    
    @property
    def num_classes(self):
        return self._num_classes
    
    @property
    def samples(self):
        return self._samples
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def get_samples_and_labels(self):
        return (self._samples, self._labels)
    
    def get_db_stats(self):
        return self._db_stats
    
    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._samples = self.samples[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._samples[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._samples = self.samples[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self.samples[start:end]
            labels_new_part = self.labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._samples[start:end], self._labels[start:end]


        
    def normalize(self, norm_type, stats_from_train_to_use_in_test=None):
        imgs, zca_instance = normalize_imgs( self._samples, norm_type, stats_from_train_to_use_in_test )
        self._samples = imgs
        self._update_stats(norm_type, zca_instance) # updates self._db_stats
        
        
    
    
    
