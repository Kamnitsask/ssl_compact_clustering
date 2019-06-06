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

import os
import sys # for checking python version
import pickle
import numpy as np
import scipy.io as sio
#from tflearn.data_utils import shuffle, to_categorical
from collections import OrderedDict

from cclp.data_utils.datasets import DataSet
from cclp.data_utils.misc_utils import makeArrayOneHot, sample_uniformly, sample_by_class

import gzip
import time

from configs import local_dirs_to_data

# constants
TRAIN_DB = 'train'
TEST_DB = 'test'

class DataManager(object):
    def __init__(self):
        self._currentBatch = None
        self.datasetsDict = OrderedDict()
        
    def _timeReading(self, readerFunction, filepath):
        LOG.info("Loading data from: "+str(filepath))
        startLoadTime = time.clock()
        dataRead = readerFunction(filepath)
        endLoadTime = time.clock()
        totalLoadTime = endLoadTime - startLoadTime
        LOG.info("[TIMING] totalLoadTime = "+str(totalLoadTime)) # svhn: train - 1.2 sec, test - 0.5, extra - 9.5 secs.
        return dataRead
        
    # Not used because we use tf functionality to do batching.
    def getCurrentBatch(self):
        return self._currentBatch
    
    def getNextBatch(self, batchSize, trainValTestStr):
        # x_value: the images for batch. [batchSize, r, c, rgb] or [batchSize, r*c*rgb] if previously asked to reshape dataset. Eg for MLP.
        # y_value: the labels for batch. Can be chosen to come as one_hot (shape [batchSize, 10]) or not ([batchSize, ]). See DataSet() for choice. 
        x_value, y_value = self.datasetsDict[trainValTestStr].next_batch(batchSize)
        self._currentBatch = (x_value, y_value)
        return self._currentBatch
    
    def getShapeOfAnImage(self, trainValTestStr):
        return self.datasetsDict[trainValTestStr].getShapeOfAnImage()
    def getShapeOfASample(self, trainValTestStr):
        return self.datasetsDict[trainValTestStr].getShapeOfASample()
    
    
    def print_characteristics_of_db(self, pause=True):
        LOG.debug("\n")
        LOG.debug("========= PRINTING CHARACTERISTICS OF THE DATABASE ==============")
        for data_str in self.datasetsDict.keys():
            imgs, lbls = self.datasetsDict[data_str].get_samples_and_labels()
            LOG.debug("["+data_str+"]-images, Shape: "+str(imgs.shape)+"| Type: "+str(imgs.dtype)+"| Min: "+str(np.min(imgs))+"| Max: "+str(np.max(imgs)))
            LOG.debug("["+data_str+"]-labels, Shape: "+str(lbls.shape)+"| Type: "+str(lbls.dtype)+"| Min: "+str(np.min(lbls))+"| Max: "+str(np.max(lbls)))
            count_class_occ = [ np.sum(lbls == class_i) for class_i in range(np.max(lbls)+1) ]
            LOG.debug("Class occurrences: "+str(count_class_occ))
        if pause:
            try:
                user_input = raw_input("Input a key to continue...")
            except:
                LOG.warn("Tried to wait and ask for user input to continue, but something went wrong. Probably nohup? Continuing...")
        LOG.debug("==========================================================================\n")
    
    
    def normalize_datasets(self, norm_type):
        train_db_normalized_already = False
        for data_str in self.datasetsDict.keys():
            stats_to_use = None
            if data_str == TRAIN_DB:
                train_db_normalized_already = True # To check that train is normed before test.
            elif data_str == TEST_DB:
                assert train_db_normalized_already
                stats_to_use = self.datasetsDict[TRAIN_DB].get_db_stats()

            self.datasetsDict[data_str].normalize( norm_type, stats_to_use )
        
    
    def sample_folds(self,
                     val_on_test,
                     num_val_samples,
                     num_lbl_samples,
                     num_unlbl_samples,
                     unlbl_overlap_val,
                     unlbl_overlap_lbl,
                     seed=None):
        # Mode 1:
        # |---validation---||---supervision---|
        #                   |------- unsupervised -------|
        # Mode 2:
        # |---validation---||---supervision---|
        # |------------------------- unsupervised -------|
        
        LOG.info("==== Sampling the folds for validation/labelled/unlabelled training ====")
        rng = np.random.RandomState(seed=seed)
        num_classes = self.datasetsDict[TRAIN_DB].num_classes
        
        # Sample VALIDATION data.
        dataset_for_val = self.datasetsDict[TEST_DB] if val_on_test else self.datasetsDict[TRAIN_DB]
        assert num_val_samples >= -1
        available_indices_val = np.asarray( range( len(dataset_for_val.samples) ), dtype="int32")
        selected_for_val = sample_uniformly(available_indices_val, num_val_samples, rng)
        val_samples = dataset_for_val.samples[ selected_for_val ]
        val_labels = dataset_for_val.labels[ selected_for_val ]
        # Debugging info for unlabelled.
        LOG.debug("[Validation] fold is test data: "+str(val_on_test)+"| Shape: Samples: "+str(val_samples.shape)+"| Labels: "+str(val_labels.shape))
        LOG.debug("[Validation] fold Class occurrences: "+str( [ np.sum(val_labels == class_i) for class_i in range(num_classes) ] ))
        
        # Sample LABELLED training data.
        # Labelled data used for supervision makes no sense to overlap with validation data in any setting.
        dataset_for_lbl = self.datasetsDict[TRAIN_DB]
        available_indices_lbl = np.asarray( range( len(dataset_for_lbl.samples) ), dtype="int32" )
        if not val_on_test: # If we are validating on subset of train data, exclude val from training.
            LOG.info("[Labelled] fold will exclude samples selected for validation.")
            available_indices_lbl = available_indices_lbl[ np.invert( np.isin(available_indices_lbl, selected_for_val) ) ]
        # Get available samples per class because we will sample per class.
        indices_of_all_lbl_samples = np.asarray( range(len(dataset_for_lbl.samples)), dtype="int32" ) # Array, to do advanced indexing 3 lines below.
        available_indices_lbl_list_by_class = []
        for c in range(num_classes):
            indices_of_lbl_samples_class_c = indices_of_all_lbl_samples[ dataset_for_lbl.labels == c ]
            available_indices_lbl_c = indices_of_lbl_samples_class_c[ np.isin(indices_of_lbl_samples_class_c, available_indices_lbl) ]
            available_indices_lbl_list_by_class.append(available_indices_lbl_c)
                
        # Sample per class.
        num_lbl_samples_per_c = num_lbl_samples // num_classes if num_lbl_samples != -1 else -1 # -1 will use all data for each class.
        # selected_for_lbl_list_by_class: Will be a list of arrays. Each array, the indices of selected samples for c.
        selected_for_lbl_list_by_class = sample_by_class( available_indices_lbl_list_by_class, num_lbl_samples_per_c, rng )
        selected_for_lbl = [item for sublist in selected_for_lbl_list_by_class for item in sublist] # flatten the by_class list of sublists.
        
        train_samples_lbl_list_by_class = [] # Will be a list of arrays. Each of the c arrays has lbled samples of class c, to train on.
        for c in range(num_classes):
            train_samples_lbl_list_by_class.append( dataset_for_lbl.samples[ selected_for_lbl_list_by_class[c] ] )
            LOG.debug("[Labelled] fold for Class ["+str(c)+"] has Shape: Samples: " + str(train_samples_lbl_list_by_class[c].shape) )
        
        # Sample UNLABELLED training data.
        dataset_for_unlbl = self.datasetsDict[TRAIN_DB]
        available_indices_unlbl = np.asarray( range( len(dataset_for_unlbl.samples) ), dtype="int32" )
        
        if not val_on_test and not unlbl_overlap_val: # If validating on train, and unlabelled should not to overlap, exclude val.
            LOG.info("[Unlabelled] fold will exclude samples selected for validation.")
            available_indices_unlbl = available_indices_unlbl[ np.invert( np.isin(available_indices_unlbl, selected_for_val) ) ]
        if not unlbl_overlap_lbl:
            LOG.info("[Unlabelled] fold will exclude samples selected as labelled.")
            available_indices_unlbl = available_indices_unlbl[ np.invert( np.isin(available_indices_unlbl, selected_for_lbl) ) ]
        selected_for_unlbl = sample_uniformly(available_indices_unlbl, num_unlbl_samples, rng)
        
        train_samples_unlbl = dataset_for_unlbl.samples[ selected_for_unlbl ]
        # Debugging info for unlabelled.
        DEBUG_train_labels_unlbl = dataset_for_unlbl.labels[ selected_for_unlbl ]
        LOG.debug("[Unlabelled] fold has Shape: Samples: "+str(train_samples_unlbl.shape)+"| Labels: "+str(DEBUG_train_labels_unlbl.shape))
        LOG.debug("[Unlabelled] fold Class occurrences: "+str( [ np.sum(DEBUG_train_labels_unlbl == class_i) for class_i in range(num_classes) ] ))
        
        LOG.info("==== Done sampling the folds ====")
        return train_samples_lbl_list_by_class, train_samples_unlbl, val_samples, val_labels
        
        
class MnistManager(DataManager):
    def __init__(self, pathToDataFolder=None, boolOneHot=False, dtypeStrX="float", reshape=False ):
        DataManager.__init__(self)
        if pathToDataFolder is None:
            pathToDataFolder = local_dirs_to_data.mnist
        pathToTrainImages = pathToDataFolder+"/train-images-idx3-ubyte.gz"
        pathToTrainLabels = pathToDataFolder+"/train-labels-idx1-ubyte.gz"
        pathToTestImages = pathToDataFolder+"/t10k-images-idx3-ubyte.gz"
        pathToTestLabels = pathToDataFolder+"/t10k-labels-idx1-ubyte.gz"
        
        (npArrTrainX, npArrTrainY) = self._readNpArrXYFromDisk( pathToTrainImages, pathToTrainLabels, boolOneHot, dtypeStrX )
        self.datasetsDict[TRAIN_DB] = DataSet(npArrTrainX, npArrTrainY, reshape=reshape)
        LOG.debug("[SHAPE] npArrTrainX.shape = "+str(npArrTrainX.shape))
        LOG.debug("[SHAPE] npArrTrainY.shape = "+str(npArrTrainY.shape))
        
        (npArrTestX, npArrTestY) = self._readNpArrXYFromDisk( pathToTestImages, pathToTestLabels, boolOneHot, dtypeStrX )
        self.datasetsDict[TEST_DB] = DataSet(npArrTestX, npArrTestY, reshape=reshape)
        LOG.debug("[SHAPE] npArrTestX.shape = "+str(npArrTestX.shape))
        LOG.debug("[SHAPE] npArrTestY.shape = "+str(npArrTestY.shape))
    
    def _readNpArrXYFromDisk(self, filepathToImages, filepathToLabels, boolOneHot, dtypeStrX):
        # X: [samples, r, c, 1] (only one channel). Values in range [0, 255]
        imagesX = self._timeReading( self.readMnistImagesFromDisk, filepathToImages )
        # Y: [samples]
        labelsY = self._timeReading( self.readMnistLabelsFromDisk, filepathToLabels )
        
        imagesX = self._preprocessX(imagesX, dtypeStrX)
        labelsY = self._preprocessY(labelsY, boolOneHot, "uint8")
        return ( imagesX, labelsY )
    
    # Save as SVHN
    def _preprocessX(self, npArrX, dtypeStr):
        # Convert from [0, 255] -> [0.0, 1.0].
        if "int" in dtypeStr:
            preprocImages = npArrX.astype(dtypeStr)
        elif "float" in dtypeStr:
            npArrX = npArrX.astype(dtypeStr)
            imagesX0To1 = np.multiply(npArrX, 1.0 / 255.0)
            preprocImages = 2 * imagesX0To1 - 1 # In the [-1, +1] bracket, just like the tanh of the G.
        else:
            raise NotImplementedError()
        return preprocImages
    
    def _preprocessY(self, npArrY, boolOneHot, dtypeStr="uint8"):
        if not "int" in dtypeStr:
            raise NotImplementedError()
        npArrY = npArrY.astype(dtypeStr)
        if boolOneHot:
            npArrY = makeArrayOneHot(npArrY, 10, 1)
        return npArrY
    
    # Adapted from tensorflow/contrib/learn/python/learn/datasets/mnist.py
    def _read32(self, bytestream):
        dt = np.dtype(np.uint32).newbyteorder('>')
        return np.frombuffer(bytestream.read(4), dtype=dt)[0]
      
    def readMnistImagesFromDisk(self, filepath):
        """Extract the images into a 4D uint8 np array [index, y, x, depth].
        Returns:
        data: A 4D uint8 np array [index, y, x, depth].
        Raises:
        ValueError: If the bytestream does not start with 2051.
        """
        with open(filepath, 'rb') as f:
            LOG.info('Extracting '+str(f.name))
            with gzip.GzipFile(fileobj=f) as bytestream:
                magic = self._read32(bytestream)
                if magic != 2051:
                    raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, f.name))
                num_images = self._read32(bytestream)
                rows = self._read32(bytestream)
                cols = self._read32(bytestream)
                buf = bytestream.read(rows * cols * num_images)
                data = np.frombuffer(buf, dtype=np.uint8)
                data = data.reshape(num_images, rows, cols, 1)
                return data

    def readMnistLabelsFromDisk(self, filepath):
        with open(filepath, 'rb') as f:
            LOG.debug('Extracting '+str(f.name))
            with gzip.GzipFile(fileobj=f) as bytestream:
                magic = self._read32(bytestream)
                if magic != 2049:
                    raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, f.name))
                num_items = self._read32(bytestream)
                buf = bytestream.read(num_items)
                labels = np.frombuffer(buf, dtype=np.uint8)
                return labels

class SvhnManager(DataManager):
    def __init__(self, pathToDataFolder=None, boolOneHot=False, dtypeStrX="float", reshape=False, loadExtra=False ):
        DataManager.__init__(self)
        if pathToDataFolder is None:
            pathToDataFolder = local_dirs_to_data.svhn
        pathToTrainMat = pathToDataFolder + "/train_32x32.mat"
        pathToTestMat = pathToDataFolder + "/test_32x32.mat"
        pathToExtraMat = pathToDataFolder + "/extra_32x32.mat"
        
        # x: type uint8, [0,255] # ~210MB. 73257 images.
        # y: type uint8. [1,10]. Zero (image) has label 10 for some reason.
        (npArrTrainX, npArrTrainY) = self._readNpArrXYFromDisk( pathToTrainMat, boolOneHot, dtypeStrX )
        self.datasetsDict[TRAIN_DB] = DataSet(npArrTrainX, npArrTrainY, reshape=reshape)
        LOG.debug("[SHAPE] npArrTrainX.shape = "+str(npArrTrainX.shape))
        LOG.debug("[SHAPE] npArrTrainY.shape = "+str(npArrTrainY.shape))
        # ~70MB
        (npArrTestX, npArrTestY) = self._readNpArrXYFromDisk( pathToTestMat, boolOneHot, dtypeStrX )
        self.datasetsDict[TEST_DB] = DataSet(npArrTestX, npArrTestY, reshape=reshape)
        LOG.debug("[SHAPE] npArrTestX.shape = "+str(npArrTestX.shape))
        LOG.debug("[SHAPE] npArrTestY.shape = "+str(npArrTestY.shape))
        # ~1.4GB
        if loadExtra:
            (npArrExtraX, npArrExtraY) = self._readNpArrXYFromDisk( pathToExtraMat, boolOneHot, dtypeStrX )
            self.datasetsDict['extra'] = DataSet(npArrExtraX, npArrExtraY, reshape=reshape)
            LOG.debug("[SHAPE] npArrExtraX.shape = "+str(npArrTestX.shape))
            LOG.debug("[SHAPE] npArrExtraY.shape = "+str(npArrExtraY.shape))
    
    def _readNpArrXYFromDisk(self, filepath, boolOneHot, dtypeStrX):
        dataDict = self._timeReading(sio.loadmat, filepath) # SVHN reads a disctionary {'X': nparray, 'y': nparray}
        # SVHN: X is read in shape [32, 32, 3, numOfSamples]. Y is read in [numOfSamples, 1]
        # Need to fix the shape of X, and take away the last dimension of Y (1), cause it doesnt correspond to how mnist is.
        imagesX = dataDict['X']
        labelsY = dataDict['y']
        
        imagesX = self._preprocessX(imagesX, dtypeStrX)
        labelsY = self._preprocessY(labelsY, boolOneHot, "uint8")
        return ( imagesX, labelsY )
    
    def _preprocessX(self, npArrX, dtypeStr):
        # Need to fix the shape of X. Which for SVHN has the numberOfSamples at last axis.
        npArrX = np.transpose(npArrX, (3,0,1,2)) # Now the shape is? [ numOfSamples, h=32, w=32, 3]
        
        # Normalize with their mean value per channel.
        #meanPerSamplePerChannel = np.mean(npArrX, axis=(1,2)) # [numOfSamples, 3 (rgb)]
        #npArrX = npArrX - meanPerSamplePerChannel[:, np.newaxis, np.newaxis, :] #np.newaxis broadcasts.
        
        # Convert from [0, 255] -> [0.0, 1.0].
        if "int" in dtypeStr:
            preprocImages = npArrX.astype(dtypeStr)
        elif "float" in dtypeStr:    
            npArrX = npArrX.astype(dtypeStr)
            imagesX0To1 = np.multiply(npArrX, 1.0 / 255.0)
            preprocImages = 2 * imagesX0To1 - 1 # In the [-1, +1] bracket, just like the tanh of the G.
        else:
            raise NotImplementedError()
        return preprocImages
        
    def _preprocessY(self, npArrY, boolOneHot, dtypeStr="uint8"):
        # In SVHN, Zero is encoded as the int 10. Change it.
        npArrY[ npArrY==10 ] = 0
        # In SVHN, labels are loaded with shape [numSamples, 1]. Get rid of last dimension.
        npArrY = npArrY.ravel()
        
        if not "int" in dtypeStr:
            raise NotImplementedError()
        npArrY = npArrY.astype(dtypeStr)
        if boolOneHot:
            npArrY = makeArrayOneHot(npArrY, 10, 1)
        return npArrY
        
        
        
        
        
class Cifar10Manager(DataManager):
    def __init__(self, pathToDataFolder=None, boolOneHot=False, dtypeStrX="float", reshape=False, loadExtra=False ):
        DataManager.__init__(self)
        if pathToDataFolder is None:
            pathToDataFolder = local_dirs_to_data.cifar10
        pathToTrainMat = pathToDataFolder + "/data_batch_"
        pathToTestMat = pathToDataFolder + "/test_batch"
        
        # x: type uint8, [0,255] # ~210MB. 73257 images.
        # y: type uint8. [1,10]. Zero (image) has label 10 for some reason.
        (npArrTrainX, npArrTrainY) = self._readNpArrXYFromDisk( pathToTrainMat, boolOneHot, dtypeStrX, train_or_test="train" )
        self.datasetsDict[TRAIN_DB] = DataSet(npArrTrainX, npArrTrainY, reshape=reshape)
        LOG.debug("=============== Data Manager ==============")
        LOG.debug("[SHAPE] npArrTrainX.shape = "+str(npArrTrainX.shape)+"\t Type: "+str(npArrTrainX.dtype))
        LOG.debug("[SHAPE] npArrTrainY.shape = "+str(npArrTrainY.shape)+"\t Type: "+str(npArrTrainY.dtype))
        # ~70MB
        (npArrTestX, npArrTestY) = self._readNpArrXYFromDisk( pathToTestMat, boolOneHot, dtypeStrX, train_or_test="test" )
        self.datasetsDict[TEST_DB] = DataSet(npArrTestX, npArrTestY, reshape=reshape)
        LOG.debug("[SHAPE] npArrTestX.shape = "+str(npArrTestX.shape)+"\t Type: "+str(npArrTestX.dtype))
        LOG.debug("[SHAPE] npArrTestY.shape = "+str(npArrTestY.shape)+"\t Type: "+str(npArrTestY.dtype))
    
    def _readNpArrXYFromDisk(self, filepath, boolOneHot, dtypeStrX, train_or_test):
        if train_or_test==TRAIN_DB:
            loaded_data = [self._unpickle( filepath + str(i) ) for i in range(1,6)]
            imagesX = np.concatenate([d['x'] for d in loaded_data],axis=0)
            labelsY = np.concatenate([d['y'] for d in loaded_data],axis=0)
        elif train_or_test==TEST_DB:
            loaded_data = self._unpickle( filepath )
            imagesX = loaded_data['x']
            labelsY = loaded_data['y']
        else:
            raise NotImplementedError('subset should be either train or test')
        
        # CIFAR10, X loaded in shape: [samples, 3, r, c].
        imagesX = self._preprocessX(imagesX, dtypeStrX)
        labelsY = self._preprocessY(labelsY, boolOneHot, "uint8")
        return ( imagesX, labelsY )
    
    def _unpickle(self, file):
        fo = open(file, 'rb')
        if sys.version_info >= (3, 0):
            d = pickle.load(fo, encoding='latin1') # without an encoding, py3 complains.
        else : # python 2
            d = pickle.load(fo)
        fo.close()
        LOG.debug("[RAW DATA] List of len: "+str(len(d['data']))+"\t Type of 1st element: "+str( type(d['data'][0]))+"\t dtype of 1st element: "+str( d['data'][0].dtype) )
        LOG.debug("[RAW LABELS] List of len: "+str( len(d['labels']) )+"\t Type  of 1st element: "+str( type(d['labels'][0])) )
        return {'x': d['data'].reshape((10000,3,32,32)), 'y': np.array(d['labels']).astype(np.uint8)}

    def _preprocessX(self, npArrX, dtypeStr):
        # Change shape of Cifar10 from [samples, 3, r, c] to [samples, r, c, 3]
        npArrX = np.transpose(npArrX, (0,2,3,1))
        
        # Convert from [0, 255] -> [-1.0, 1.0].
        if "int" in dtypeStr:
            preprocImages = npArrX.astype(dtypeStr)
        elif "float" in dtypeStr:
            raise NotImplementedError()
            
            #npArrX = npArrX.astype(dtypeStr)
            #imagesX0To1 = np.multiply(npArrX, 1.0 / 255.0)
            #preprocImages = 2 * imagesX0To1 - 1 # In the [-1, +1] bracket, just like the tanh of the G.
        else:
            raise NotImplementedError()
        return preprocImages
        
    def _preprocessY(self, npArrY, boolOneHot, dtypeStr="uint8"):
        
        if not "int" in dtypeStr:
            raise NotImplementedError()
        npArrY = npArrY.astype(dtypeStr)
        if boolOneHot:
            npArrY = makeArrayOneHot(npArrY, 10, 1)
        return npArrY
    
    
    
    
    
        
        
        