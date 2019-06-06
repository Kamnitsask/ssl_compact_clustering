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

import sys
import os
from functools import partial

import numpy as np

import tensorflow as tf
from tensorflow.python.training import saver as tf_saver

from cclp.sampling.preprocessing import normalize_tens
from cclp.sampling.augmentation import augment_tens
from cclp.data_utils import data_managers
from cclp.neuralnet.models.feat_extractor_archs import get_feat_extractor_z_func
from cclp.embeddings.visualise_embeddings import plot_2d_emb
from cclp.frontend.logging.utils import datetime_str as timestr
from cclp.frontend.logging import metrics
from cclp.sampling import samplers
from cclp.neuralnet.models import classifier
from cclp.neuralnet.trainers import trainers


def train( sessionNModelFlags, trainerFlags ):
    tf.logging.set_verbosity(tf.logging.WARN) # set to INFO for more exhaustive.
    config = tf.ConfigProto(log_device_placement = False)
    config.gpu_options.allow_growth = False
    
    # Load data.
    if sessionNModelFlags["dataset"] == 'mnist':
        db_manager = data_managers.MnistManager(dtypeStrX="uint8")
    elif sessionNModelFlags["dataset"] == 'svhn':
        db_manager = data_managers.SvhnManager(dtypeStrX="uint8")
    elif sessionNModelFlags["dataset"] == 'cifar':
        db_manager = data_managers.Cifar10Manager(dtypeStrX="uint8")
        
    LOG.info("")
    LOG.info("Before pre-augmentation normalization...")
    db_manager.print_characteristics_of_db(pause=False)
    # Data normalization
    db_manager.normalize_datasets(sessionNModelFlags["norm_imgs_pre_aug"])
    
    LOG.info("")
    LOG.info("After pre-augmentation normalization...")
    # train/val/unlab images in tensor of shape: [batch, x, y, channels]
    db_manager.print_characteristics_of_db(pause=False)
    
    # Sample Validation, labeled and unlabeled training data.
    seed = sessionNModelFlags["seed"]
    (train_samples_lbl_list_by_class,
    train_samples_unlbl,
    val_samples, val_labels) = db_manager.sample_folds(sessionNModelFlags['val_on_test'],
                                                       sessionNModelFlags['num_val_samples'],
                                                       sessionNModelFlags['num_lbl_samples'],
                                                       sessionNModelFlags['num_unlbl_samples'],
                                                       sessionNModelFlags['unlbl_overlap_val'],
                                                       sessionNModelFlags['unlbl_overlap_lbl'],
                                                       seed=seed)
    
    num_classes = db_manager.datasetsDict['train'].num_classes
    image_shape = db_manager.datasetsDict['train'].getShapeOfAnImage() # [x, y, channs]. Without the batch size.
    str_val_or_test = "Test" if sessionNModelFlags['val_on_test'] else "Validation"
    
    # Up until here, everything is numpy. Tensorflow starts below:
    graph = tf.Graph()
    with graph.as_default():
        with tf.device(tf.train.replica_device_setter(ps_tasks=0, merge_devices=True)):
            LOG.info("==========================================================")
            LOG.info("================== Creating the graph ====================")
            LOG.info("==========================================================\n")
            
            # Get batch. Tensorflow batch-constructors. They return tensorflow tensors. Shape: (batch, H, W, Channs)
            t_sup_images, t_sup_labels = samplers.get_tf_batch_with_n_samples_per_class( train_samples_lbl_list_by_class, sessionNModelFlags["n_lbl_per_class_per_batch"], seed )
            t_unsup_images, _ = samplers.get_tf_batch(train_samples_unlbl, None, sessionNModelFlags["n_unlbl_per_batch"], seed)
            
            # Partially define augmentation, renormalization and the feature extractor functions...
            # ... So that they can later be applied on an input tensor straightforwardly.
            # Apply augmentation
            is_train_db_zero_centered = db_manager.datasetsDict['train'].get_db_stats() # to augment is accordingly.
            augment_tens_func = partial(augment_tens, params=sessionNModelFlags["augm_params"], db_zero_centered=is_train_db_zero_centered)
            # Re-normalize after augmentation if wanted.
            normalize_tens_func = partial(normalize_tens, norm_type=sessionNModelFlags["norm_tens_post_aug"])
            # Create function that the feature extractor z(.)
            embed_tens_func = partial( get_feat_extractor_z_func(sessionNModelFlags["feat_extractor_z"]), # function pointer.
                                       emb_z_size=sessionNModelFlags["emb_z_size"],
                                       emb_z_act=sessionNModelFlags["emb_z_act"],
                                       l2_weight=sessionNModelFlags["l2_reg_feat"],
                                       batch_norm_decay=sessionNModelFlags["batch_norm_decay"],
                                       seed=seed )
            
            # Initialize the model class (it does not yet build tf computation graph.)
            model = classifier.Classifier( augment_tens_func,
                                           normalize_tens_func,
                                           embed_tens_func,
                                           num_classes,
                                           image_shape,
                                           sessionNModelFlags["l2_reg_classif"],
                                           sessionNModelFlags["classifier_dropout"] )
            
            # Build the tf-graph for the forward pass through feature extractor z and classifier.
            model.forward_pass(input=None, tensor_family="eval", is_training=False, init_vars=True, seed=seed, image_summary=False)
            summary_op_eval = tf.summary.merge_all(key="eval_summaries")
            model.forward_pass(input=t_sup_images, tensor_family="train_sup", is_training=True, init_vars=False, seed=seed, image_summary=False)
            model.forward_pass(input=t_unsup_images, tensor_family="train_unsup", is_training=True, init_vars=False, seed=seed, image_summary=False)
            
            # Create trainer and make the op. (I should do something so concise for the setup_model too.)
            trainer = trainers.Trainer( params = trainerFlags, net_model=model, t_sup_labels=t_sup_labels )
            train_op = trainer.get_train_op()
            increase_model_step_op = trainer.get_increase_model_step_op()
            
            saver = tf_saver.Saver(max_to_keep=sessionNModelFlags["max_checkpoints_to_keep"])
            
            summary_op = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES) # This is what makes the tensorboard's metrics.
            summary_writer = tf.summary.FileWriter( sessionNModelFlags["logdir"]["summariesTf"], graph ) # creates the subfolder if not there already.
            
            
    with tf.Session( graph=graph, config=config ) as sess:
        LOG.info(".......................................................................")
        if sessionNModelFlags['model_to_load'] is not None:
            chkpt_fname = tf.train.latest_checkpoint( sessionNModelFlags['model_to_load'] ) if os.path.isdir( sessionNModelFlags['model_to_load'] ) else sessionNModelFlags['model_to_load'] 
            LOG.info("Loading model from: "+str(chkpt_fname))
            saver.restore(sess, chkpt_fname)
        else:
            LOG.info("Initializing model...")
            tf.global_variables_initializer().run()
        LOG.info(".......................................................................\n")
        
        coordinator = tf.train.Coordinator() # Coordinates the threads. Starts and stops them.
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator) # Start the slicers/batchers registered in cclp.sampling.samplers.
        
        LOG.info("====================================================================")
        LOG.info("================== Starting training iterations ====================")
        LOG.info("====================================================================\n")
        model_step = model.get_t_step().eval(session=sess)
        while model_step < sessionNModelFlags["max_iters"] :
            
            (_,
            summaries,
            emb_train_sup,
            emb_train_unsup,
            nparr_t_sup_labels) = sess.run([ train_op,
                                            summary_op,
                                            model.tensor_families["train_sup"]["emb_z_tens"],
                                            model.tensor_families["train_unsup"]["emb_z_tens"],
                                            t_sup_labels ]) # This is the main training step.
            
            model_step = sess.run( increase_model_step_op )
            
            # The two following external if's are the same and could be merged.
            # Save summaries, create visualization of embedding if requested.
            if (model_step) % sessionNModelFlags["val_interval"] == 0 or model_step == 100:
                LOG.info('Step: %d' % model_step)
                summary_writer.add_summary(summaries, model_step) # log TRAINING metrics.
                
                
                # PLOT AND SAVE EMBEDDING
                if sessionNModelFlags["plot_save_emb"] > 0 :
                    plot_2d_emb( emb_train_sup, emb_train_unsup, nparr_t_sup_labels, train_step=model_step,
                                 save_emb=sessionNModelFlags["plot_save_emb"] in [1,3], plot_emb=sessionNModelFlags["plot_save_emb"] in [2,3], output_folder=sessionNModelFlags["logdir"]["emb"] )
                    
                    
                # ACCURACY ON TRAINING DATA.
                train_images_for_metrics = np.concatenate( train_samples_lbl_list_by_class, axis=0 )
                num_samples_per_class_train_to_eval = train_samples_lbl_list_by_class[0].shape[0]
                train_gt_lbls_for_metrics = []
                for c in range(0, num_classes):
                    train_gt_lbls_for_metrics += [c] * num_samples_per_class_train_to_eval
                train_gt_lbls_for_metrics = np.asarray(train_gt_lbls_for_metrics)
                
                train_pred_lbls_for_metrics = []
                for i in range(0, len(train_images_for_metrics), sessionNModelFlags["eval_batch_size"]):
                    [train_pred_logits_batch] =  sess.run( [ model.tensor_families["eval"]["logits_tens"] ],
                                                            { model.tensor_families["eval"]["inp_tens"] : train_images_for_metrics[ i : i+sessionNModelFlags["eval_batch_size"] ] } )
                    train_pred_lbls_for_metrics.append( train_pred_logits_batch.argmax(-1) ) # Take the classes with argmax.
                    
                train_pred_lbls_for_metrics = np.concatenate(train_pred_lbls_for_metrics)
                if train_pred_lbls_for_metrics.shape[0] > train_gt_lbls_for_metrics.shape[0]: # can happen when superv data = -1. 59230 total data, batcher fills the last batches and the whole thing returns 60000? Not sure.
                    train_pred_lbls_for_metrics = train_pred_lbls_for_metrics[ : train_gt_lbls_for_metrics.shape[0] ]
                    
                train_err = (train_gt_lbls_for_metrics != train_pred_lbls_for_metrics).mean() * 100 # What is reported is percentage.
                train_summary = tf.Summary( value=[ tf.Summary.Value( tag='Train Err', simple_value=train_err) ] )
                summary_writer.add_summary(train_summary, model_step)
                
                confusion_mtx = metrics.confusion_mtx(train_gt_lbls_for_metrics, train_pred_lbls_for_metrics, num_classes)
                LOG.info("\n"+str(confusion_mtx))
                LOG.info('Training error: %.2f %% \n' % train_err)
                
                
                # ACCURACY ON VALIDATION DATA.
                if sessionNModelFlags["val_during_train"]:
                    
                    eval_pred_lbls = [] # list of embeddings for each val batch: [ [batchSize, 10], .... [bs, 10] ]
                    for i in range(0, len(val_samples), sessionNModelFlags["eval_batch_size"]):
                        
                        [eval_pred_logits_batch,
                        summaries_eval] =  sess.run( [ model.tensor_families["eval"]["logits_tens"], summary_op_eval ],
                                                    { model.tensor_families["eval"]["inp_tens"] : val_samples[ i : i+sessionNModelFlags["eval_batch_size"] ] } )
                        eval_pred_lbls.append( eval_pred_logits_batch.argmax(-1) ) # Take the classes with argmax.
                        
                    eval_pred_lbls = np.concatenate(eval_pred_lbls) # from list to array.
                    eval_err = (val_labels != eval_pred_lbls).mean() * 100 # report percentage.
                    eval_summary = tf.Summary( value=[ tf.Summary.Value( tag=str_val_or_test+' Err', simple_value=eval_err) ] )
                    summary_writer.add_summary(eval_summary, model_step)
                    summary_writer.add_summary(summaries_eval, model_step)
                    
                    confusion_mtx = metrics.confusion_mtx(val_labels, eval_pred_lbls, num_classes)
                    LOG.info("\n"+str(confusion_mtx))
                    LOG.info(str_val_or_test+' error: %.2f %% \n' % eval_err)
                    
                    
                # SAVE MODEL
                filepath_to_save = os.path.join(sessionNModelFlags["logdir"]["trainTf"], "model-"+str(model_step)+"-"+timestr() )
                LOG.info("[SAVE] Saving model at: "+str(filepath_to_save))
                saver.save( sess, filepath_to_save, global_step=None, write_meta_graph=False, write_state=True ) # creates the subfolder if doesnt exist. Appends step if given.
            
            
        coordinator.request_stop() # Stop the threads.
        coordinator.join(threads)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
