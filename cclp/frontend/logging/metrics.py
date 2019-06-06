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

import numpy as np

def confusion_mtx(gt_lbls, pred_lbls, num_classes):
    # gt_lbls: ground truth labels. Vector, one int (class) per sample.
    # pred_lbls: ditto.
    row_per_gt_class = []
    for c in range(num_classes):
        pred_lbls_for_samples_of_this_class = pred_lbls[gt_lbls == c]
        row = np.bincount(pred_lbls_for_samples_of_this_class, minlength=num_classes) # returns a vector with shape (num_classes,).
        row = np.reshape(row, newshape=(1, num_classes)) # now it's a (1,10) row.
        row_per_gt_class.append(row)
    
    confusion_mtx = np.concatenate( row_per_gt_class, axis=0 )
    return confusion_mtx










