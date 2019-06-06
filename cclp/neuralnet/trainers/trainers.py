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

from cclp.routines.schedules.schedules import apply_growth
from cclp.neuralnet.trainers import losses


class Trainer(object):
    # A class separate than the model, to keep separately the optimization state.
    def __init__(self, params, net_model, t_sup_labels):
        self._params = params # A dictionary or dictionary-like ConfigFlags.
        self._ema = tf.train.ExponentialMovingAverage(decay=0.99)
                
        self._loss_total_weighted = self._setup_losses( net_model, t_sup_labels )
        self._t_learning_rate = self._get_t_learning_rate( net_model ) # Can be returning scalar or tensor (eg from schedule).
        self._train_op = self._create_train_op()
        self._increase_model_step_op = tf.assign( net_model.get_t_step(), net_model.get_t_step() + 1)
        
        tf.summary.scalar( 'Loss_Total_weighted', self._loss_total_weighted )
        tf.summary.scalar( 'Learning_Rate', self._t_learning_rate )
    
    def _setup_losses(self, net_model, t_sup_labels):
        # net_model: Instance of ./cclp/neuralnet/models/classifier/Classifier.
        
        losses.add_softmax_cross_entr( logits = net_model.tensor_families["train_sup"]["logits_tens"],
                                       lbls = t_sup_labels,
                                       weight = self._params["logit_weight"] )
            
        if self._params["cc_loss_on"]:
            losses.add_cclp_loss(
                Z_l = net_model.tensor_families["train_sup"]["emb_z_tens"],
                Z_u = net_model.tensor_families["train_unsup"]["emb_z_tens"],
                y_l_lbls = t_sup_labels,
                c_classes = net_model.get_num_classes(),
                # Params for creating the graph
                sim_metric = self._params["cc_sim_metric"],
                l2_sigmas = self._params["cc_l2_sigmas_init"],
                l2_sigmas_trainable = self._params["cc_l2_sigmas_trainable"],
                # Params for CCLP loss
                cclp_weight = self._params["cc_weight"],
                cclp_steps = self._params["cc_steps"],
                sum_over_chains = self._params["cc_sum_over_chains"],
                # Others
                e_smooth = self._params["cc_e_smooth"],
                optim_smooth_mtx = self._params["cc_optim_smooth_mtx"] )
            
            
        loss_total_weighted = tf.losses.get_total_loss(add_regularization_losses=True) #  tf keeps track of everything. Losses registered eg in add_logit_loss and L2 are here.
        return loss_total_weighted
    
    
    def _get_t_learning_rate(self, net_model):
        # Set up learning rate
        if self._params["lr_sched_type"] == 'expon_decay':
            t_learning_rate = tf.maximum( tf.train.exponential_decay( self._params["lr_expon_init"], net_model.get_t_step(), self._params["lr_expon_decay_steps"], self._params["lr_expon_decay_factor"], staircase=True),
                                          self._params["lr_min_value"])
        elif self._params["lr_sched_type"] == 'piecewise':
            # In github it was said that piecewise was used for svhn.
            t_learning_rate = tf.maximum( tf.train.piecewise_constant( net_model.get_t_step(), boundaries = [ tf.cast(v, tf.int32) for v in self._params["lr_piecewise_boundaries"] ], values = self._params["lr_piecewise_values"] ),
                                        self._params["lr_min_value"])
            
        return t_learning_rate
    
    
    def _get_grads_after_calc_grads_and_g_to_v_per_loss(self, list_of_trainable_vars):
        
        # Get all losses
        list_of_all_added_losses = tf.losses.get_losses() # THIS DOES NOT INCLUDE REGULARIZATION LOSSES!
        # ... See last line: https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/python/ops/losses/util.py
        list_of_all_added_losses += tf.get_collection( tf.GraphKeys.REGULARIZATION_LOSSES ) # This is where L2 is placed.
        
        LOG.debug("list_of_all_added_losses = " + str(list_of_all_added_losses) )
        LOG.debug("tf.get_collection( tf.GraphKeys.REGULARIZATION_LOSSES ) = " + str(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) )
        list_of_grads_for_each_var_per_loss = [] # will be of shape NumLosses x numVars
        for loss in list_of_all_added_losses:
            LOG.info('Computing grads of each var wrt Loss: '+loss.name )
            grads_for_this_loss = tf.gradients( loss, list_of_trainable_vars )
            list_of_grads_for_each_var_per_loss.append( grads_for_this_loss )
            
        
        # Now that you have for each variable, the gradients from the different losses separately, compute the ratios to the variable's value, and an ema to report.
        list_of_loss_names_to_print_ratios = ['loss_logit_weighted', 'loss_LP_unl_entr_weighted', 'loss_hop0_weighted',
                                               'loss_hop1_weighted', 'loss_hop2_weighted', 'loss_hop3_weighted', 'loss_hop4_weighted', 'loss_hop5_weighted',
                                               'loss_hopNeg0_weighted' ]
        list_of_ema_update_ops = []
        for loss_i in range( len(list_of_all_added_losses) ) :
            this_loss_name = list_of_all_added_losses[loss_i].name
            if any( [ this_loss_name.startswith( name_of_interest ) for name_of_interest in list_of_loss_names_to_print_ratios ] ):
                LOG.debug('LOSS FOUND! this_loss_name='+this_loss_name)
                
                grads_for_this_loss = list_of_grads_for_each_var_per_loss[loss_i]
                
                sum_of_all_pow2_grads = 0
                sum_of_all_pow2_vars = 0
                for grad, var in zip( grads_for_this_loss, list_of_trainable_vars ):
                    # Each "grad" is of different shape. eg a tensor of shape [3,3,32,32] for conv, or [3] for bias, etc. So I need to treat them carefully.
                    # Same for Variables tensors.
                    if grad is None:
                        continue # eg in the case that a var does not depend on a loss. eg classif layer to auxiliary losses.
                    
                    sum_of_all_pow2_grads += tf.reduce_sum( tf.pow(grad, 2) )
                    sum_of_all_pow2_vars += tf.reduce_sum( tf.pow(var, 2) )
                norm_grads = tf.sqrt( sum_of_all_pow2_grads )
                norm_vars = tf.sqrt( sum_of_all_pow2_vars )
                ratio_g_to_v = norm_grads / norm_vars
                
                # Maintain and report a moving average for each ratio:
                list_of_ema_update_ops.append( self._ema.apply([ratio_g_to_v]) )
                ema_ratio_g_to_v = self._ema.average( ratio_g_to_v )
                tf.summary.scalar('RatioGtoV_'+this_loss_name, ema_ratio_g_to_v)
                
                
        # Add up the gradients from each different loss into one total gradient for each variable, that the optimizer will then apply
        grads_total_for_each_var = None
        for grads_wrt_specific_loss in list_of_grads_for_each_var_per_loss:
            if grads_total_for_each_var is None:
                grads_total_for_each_var = grads_wrt_specific_loss
            else:
                assert len(grads_total_for_each_var) == len(grads_wrt_specific_loss)
                num_var_n_grad_tensors = len(grads_total_for_each_var)
                for grad_i in range( num_var_n_grad_tensors ):
                    if grads_wrt_specific_loss[grad_i] is None:
                        continue # eg if a loss does not depend on a variable. Eg, LP losses wrt classification layer.
                    elif grads_total_for_each_var[grad_i] is None: # eg if the corresponding variable was independent of the very first loss.
                        grads_total_for_each_var[grad_i] = grads_wrt_specific_loss[grad_i]
                    else:
                        grads_total_for_each_var[grad_i] = grads_total_for_each_var[grad_i] + grads_wrt_specific_loss[grad_i]
                        
        
        return grads_total_for_each_var, list_of_ema_update_ops
    
    
    def _create_train_op(self):
        
        list_of_optimizers = []
        list_of_trainable_var_collections = []
        list_of_train_ops = []
        
        """
        LOG.debug("***** Are we correctly getting update ops of BN? *****" )
        LOG.debug("tf.get_collection(tf.GraphKeys.UPDATE_OPS)=" + str( tf.get_collection(tf.GraphKeys.UPDATE_OPS) ) )
        LOG.debug("len( tf.get_collection(tf.GraphKeys.UPDATE_OPS) ) = " + str( len(tf.get_collection(tf.GraphKeys.UPDATE_OPS)) ) )
        for thing in tf.get_collection(tf.GraphKeys.UPDATE_OPS):
            LOG.debug( "thing = " + str(thing) )
        """
        
        # Make main op, training all the tf.GraphKeys.TRAINABLE_VARIABLES. All separately trained are in different collections.
        trainable_vars_main = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES )
        list_of_trainable_var_collections.append( trainable_vars_main ) # concatente all trainable vars in a list/collection.
        
        optimizer_main = tf.train.AdamOptimizer( self._t_learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-07 )
        #optimizer_main = tf.train.RMSPropOptimizer( self._t_learning_rate, decay=0.9, momentum=0.6, epsilon=1e-8 )
        #optimizer_main = tf.train.MomentumOptimizer( self._t_learning_rate, momentum=0.9, use_nesterov=True )
        
        list_of_optimizers.append( optimizer_main )
        
        if self._params["cc_loss_on"] and self._params["cc_sim_metric"] == "L2" and self._params['cc_l2_sigmas_trainable']:
            trainable_lp_l2_sigmas = tf.get_collection( 'TRAINABLE_LP_L2_SIGMAS' )
            list_of_trainable_var_collections.append( trainable_lp_l2_sigmas )
            optimizer_sigmas = tf.train.AdamOptimizer( self._t_learning_rate * self._params['cc_l2_sigmas_lr_multipl'] )
            list_of_optimizers.append( optimizer_sigmas )
        # Add more "special" trainable var collections here if needed...
        
        # Get all trainable vars in one list
        list_of_trainable_vars = [ var for sublist in list_of_trainable_var_collections for var in sublist ]
        
        if self._params["track_ratio_g_v"] :
            LOG.debug("Going to calculate grads per loss separately, to track ratio of grads/var. Slow." )
            # grads_total_for_each_var: total gradient for each variable. shape: number of variables.
            # list_of_ema_update_ops: one for each tracked loss in list_of_loss_names_to_print_ratios
            grads_total_for_each_var, list_of_ema_update_ops = self._get_grads_after_calc_grads_and_g_to_v_per_loss(list_of_trainable_vars)
            
        else :
            LOG.debug("Not tracking grads/var. Calc grad from total_loss." )
            
            grads_total_for_each_var = tf.gradients( self._loss_total_weighted, list_of_trainable_vars )
            list_of_ema_update_ops = []
            
        
        # Now lets apply the grads to the parameters, with the appropriate optimiser / learningRate.
        low_counter = 0
        for i in range( len(list_of_trainable_var_collections) ) :
            var_collection = list_of_trainable_var_collections[i]
            high_counter = low_counter+len(var_collection)
            grads_for_this_var_collection = grads_total_for_each_var[ low_counter: high_counter ]
            optimizer = list_of_optimizers[i]
            train_op = optimizer.apply_gradients( zip(grads_for_this_var_collection, var_collection) )
            list_of_train_ops.append(train_op)
            
            low_counter = high_counter
            
        all_ops_to_run_at_one_train_step = list_of_train_ops
        all_ops_to_run_at_one_train_step += list_of_ema_update_ops
        all_ops_to_run_at_one_train_step += tf.get_collection(tf.GraphKeys.UPDATE_OPS) # This one keeps updates of Batch normalization.
        total_train_op = tf.group( *all_ops_to_run_at_one_train_step )
        
        return total_train_op
    
    def get_train_op(self):
        return self._train_op
    
    def get_increase_model_step_op(self):
        return self._increase_model_step_op















