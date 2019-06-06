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


def makeArrayOneHot(arrayWithClassInts, cardinality, axisInResultToUseForCategories):
    # arrayWithClassInts: np array of shape [batchSize, r, c, z], with sampled ints expressing classes for each of the samples.
    oneHotsArray = np.zeros( [cardinality] + list(arrayWithClassInts.shape), dtype=np.float32 )
    oneHotsArray = np.reshape(oneHotsArray, newshape=(cardinality, -1)) #Flatten all dimensions except the first.
    arrayWithClassInts = np.reshape(arrayWithClassInts, newshape=-1) # Flatten
    
    oneHotsArray[arrayWithClassInts, range(oneHotsArray.shape[1])] = 1
    oneHotsArray = np.reshape(oneHotsArray, newshape=[cardinality] + list(arrayWithClassInts.shape)) # CAREFUL! cardinality first!
    oneHotsArray = np.swapaxes(oneHotsArray, 0, axisInResultToUseForCategories) # in my implementation, axisInResultToUseForCategories == 1 usually.
    
    return oneHotsArray
    
    
def sample_uniformly(array_to_sample, num_samples, rng):
    if num_samples == -1: # return all elements.
        selected_items = array_to_sample
    else:
        selected_items = rng.choice(a=array_to_sample, size=num_samples, replace=False)
    return selected_items


def sample_by_class( list_of_arrays_to_sample_by_class, num_samples_per_c, rng ):
    # num_samples_per_c: a single integer. Not a list of integers.
    # list_of_arrays_to_sample_by_class: list with number_of_classes arrays. Each array gives the indices to sample from, for class c.
    selected_items_list_by_class = [] # Will be a list of sublists. Each sublist, the indices of selected samples for c.
    num_classes = len(list_of_arrays_to_sample_by_class)
    for c in range(num_classes):
        selected_items_c = sample_uniformly(list_of_arrays_to_sample_by_class[c], num_samples_per_c, rng)
        selected_items_list_by_class.append(selected_items_c)
    
    return selected_items_list_by_class



        
        