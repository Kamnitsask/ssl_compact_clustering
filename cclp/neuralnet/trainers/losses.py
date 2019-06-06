#!/usr/bin/env python

# Copyright (c) 2018, Konstantinos Kamnitsas
#
# This program is free software; you can redistribute and/or modify
# it under the terms of the Apache License, Version 2.0. See the 
# accompanying LICENSE file or read the terms at:
# http://www.apache.org/licenses/LICENSE-2.0

from __future__ import absolute_import, division, print_function

import logging
# LOG = logging.getLogger('main') # Commented out so that file is standalone.

import numpy as np

import tensorflow as tf


#####################################################################################
#            CCLP loss: Compact Clustering via Label Propagation                    #
#####################################################################################
# Main function for our work is: add_cclp_loss(...)

def make_graph_and_transition_mtx( Z_lu, sim_metric = "dot", l2_sigmas = 1.0, l2_sigmas_trainable = False ):
    # Computes transition matrix of a fully connected graph over embeddings Z_lu of samples in a batch.
    # Z_lu: mtx shape [ num_samples (N), number-of-features-in-z (F) ]. Embeddings of samples in batch. The nodes of the graph.
    # sim_metric: 'dot' or 'L2'. Similarity metric for creating affinity (similarity) matrix of the graph.
    # ... Note that dot produce is similarity metrix. L2 is distance. -L2/sigma is the corresponding similarity metric.
    # l2_sigmas: std for scaling L2 distance. Can be int, or list of ints (one per feature of Z embedding).
    # l2_sigmas_trainable: Whether to learn sigmas during training.
    
    if sim_metric == "dot": # dot product as a similarity metric.
        aff_mtx = tf.matmul(Z_lu, Z_lu, transpose_b=True, name='dot_aff_mtx') # Graph's affinity mtx. [N,F] x [F,N] = [N, N].
        # Softmax instead of tf.exp / tf.reduce_sum( ), because when dotprod goes >100 (and it does), ...
        # ... tf.exp goes to inf. Softmax implementation handles it.
        H_trans = tf.nn.softmax(aff_mtx, name='H_trans')
        
    elif sim_metric == "L2":
        num_feats = Z_lu.get_shape()[-1] # Number of features of latent space Z.
        assert isinstance( l2_sigmas_trainable, int ) or ( len(l2_sigmas_trainable) == num_feats )
        sigmas_per_feat = np.ones( num_feats, dtype="float32" ) * np.asarray( l2_sigmas, "float32" )
        with tf.variable_scope('net', reuse=False):
            t_sigmas_per_feat = tf.Variable( sigmas_per_feat, name='sigmas_per_feat', dtype=tf.float32, trainable=False )
            if l2_sigmas_trainable:
                tf.add_to_collection( 'TRAINABLE_LP_L2_SIGMAS', t_sigmas_per_feat )
            
        Z_lu_1 = tf.expand_dims( Z_lu, axis=1 ) # adds a unary axis, turning [N, F] to [N, 1, F]
        Z_lu_2 = tf.expand_dims( Z_lu, axis=0 ) # adds a unary axis, turning [N, F] to [1, N, F]
        # When the two above will be subtracted, the 1 dimensions will be broadcasted, giving a [N, N, F] array.
        # Implementing eq15 of LP Thesis 2002.
        dist = Z_lu_1 - Z_lu_2
        dist = dist ** 2 # square of per-feats distance
        dist = dist / tf.reshape( t_sigmas_per_feat**2, [1,1,-1] ) # add 2 dims to sigmas, to broadcast them.
        dist = tf.reduce_sum( dist , axis=2 ) # L2 distance, weighted with a sigma per dimension
        aff_mtx = tf.exp( - dist ) # Affinity mtx. Symmetrical.
        
        # Transition mtx. Eq 2.2 LP Thesis 2002. Probability transiting from i (row) to j (col). Row-normalized, not col. non-symmetric.
        H_trans = aff_mtx / tf.reduce_sum( aff_mtx, axis=1, keep_dims=True ) # Normalize each row. Keep dim to broadcast.
        tf.summary.scalar('b_lp_l2_sigma_min', tf.reduce_min(t_sigmas_per_feat))
        tf.summary.scalar('b_lp_l2_sigma_max', tf.reduce_max(t_sigmas_per_feat))
        
    else:
        raise NotImplementedError('method calc_adj_matrix in \'{}\''.format(type(self).__name__))
    
    return H_trans


def label_propagation_to_unlabelled(Y_l, H_trans):
    # Calculate closed form solution of Label Propagation (LP), the class posteriors for unlabelled samples.
    # Given by Eq. 5 of CCLP paper. (Eq. 2.11 in original LP Thesis 2002).
    # Y_L: Matrix [number_of_labelled_samples (n_l), C_classes]. Labels of labelled samples in batch, one-hot form.
    # H_trans: shape [num_of_all_samples, num_of_all_samples]. Transition matrix of graph over all samples in batch.
    #      ... Rows and Columns should be ordered so that first entries are labelled samples. See Eq.4 in CCLP paper.
    #      ... Inverse of H is expected to exist. E.g. smooth matrix in advance.
    # Returns: Phi_u, the class-posteriors for the unlabelled samples estimated by LP.
    
    n_l = Y_l.get_shape()[0] # Number of labelled samples in batch.
    
    H_uu = H_trans[ n_l : , n_l : ] # Eq.4. Transition probabilities between Unlabelled samples only.
    H_ul = H_trans[ n_l : , : n_l ] # Eq.4. Transition probabilities from Unlabelled to Labelled samples only.
    I_uu = tf.eye( num_rows = int(H_uu.get_shape()[0]), num_columns = int(H_uu.get_shape()[1]), dtype = tf.float32 )
    inv_Iuu_minus_Huu = tf.matrix_inverse( I_uu - H_uu )
    Hul_mm_Yl = tf.matmul(H_ul, Y_l)
    Phi_u = tf.matmul( inv_Iuu_minus_Huu, Hul_mm_Yl ) # Eq.5. Class posteriors for unlabelled samples by LP. shape: [n_u, c_classes]
    return Phi_u


def regularize_for_compact_clustering( H_trans,
                                       Phi_lu,
                                       cclp_weight = 1.0,
                                       cclp_steps = 3,
                                       sum_over_chains = True,
                                       eps_log = 1e-6):
    # This computes a loss such that optimization drives the net to embed samples in feature space z such that ...
    # a single compact cluster is formed per class. To do this, given current class-posterior estimation by LP (PHi_lu), ...
    # it first computes the *ideal* transition matrix T (as it should be if a single compact cluster per class ...
    # was formed). It then optimizes current transition matrix H to change like the ideal. SGD optimization is ... 
    # done along markov chains of multiple lengths (steps), to preserve existing clusters during optimization (see paper).
    # H_trans: matrix [ N x N ] where N = number of all samples in batch. The current transition matrix of the graph.
    #          Elements should be ordered such that first rows/cols correspond to labelled samples, followed by unlabelled.
    # Phi_lu: matrix [ N x C ], where C = number of classes. Class posteriors by label propagation (true labels for labelled samples).
    #         Elements should be ordered such that first rows correspond to labelled samples, followed by unlabelled.
    # cclp_weight: Float. The weight w in: Loss_total = Cross_Entr + w * Loss_cclp
    # cclp_steps: Number of steps of the longest markov chain along which to optimize the path between samples.
    # sum_over_chains: True/False. If False, only the longest chain is optimized.
    assert cclp_steps >= 0
    
    # ---------- Calculate the ideal (target) transition matrix T -------------
    # Equation 6 of CCLP paper.
    # In the ideal case where a single, compact cluster per class has been formed, then ...
    # ... the probability of transiting from sample i to j should be uniform for each j of the same class as i, and zero otherwise.
    # This will be reflected in T_trans, the ideal (target) transition matrix.
    mass_per_c = tf.reduce_sum(Phi_lu, axis=0, keep_dims=True) # Mass of each class. Matrix [ 1, C_classes ]
    # phi_div_mass[i,j] interepreted as: Prob to transit to sample i from sample of class c, assuming uniform prob transiting to each sample of class c.
    phi_div_mass = Phi_lu / ( mass_per_c + 1e-5 ) # elem wise after broadcasting mass_per_c. Returns [NxC]. Mass >> eps.
    # Ideal trans mtx T[i,j]: Prob transitioning from sample i to j, assuming uniform prob to transit to any sample of same class, zero to other classes.
    # In other words, how much their class posteriors agree, normalized by the mass in the given class.
    T_trans = tf.matmul( Phi_lu, phi_div_mass, transpose_b=True, name='T_trans')
    
    
    # Compute the multiplier that will be the weight of each chain of CCLP.
    if cclp_steps > 0:
        if sum_over_chains: # Standard form of CCLP, normalize with number of chains. Eq.9.
            cclp_weight_per_chain = 1. / (cclp_steps) * cclp_weight
        else : # For ablation study (not in paper), when only the longest chain is optimized.
            cclp_weight_per_chain = cclp_weight
    else :
        cclp_weight_per_chain = 0.0
    
    # --------------- Compute 1-step CCLP loss ---------------------------
    loss_cclp = 0 # Will hold final result, Eq.9
    H_step = H_trans # H_step will hold H^(s) of Eq.8.
    loss_step = tf.reduce_mean( - tf.reduce_sum( T_trans * tf.log( H_step + eps_log), axis=[1] ) )
    tf.summary.scalar('Loss_step1', loss_step)
    if cclp_steps == 1 or sum_over_chains :
        loss_cclp += loss_step
        if cclp_weight_per_chain > 0 :
            loss_step_weighted = tf.multiply( loss_step, cclp_weight_per_chain, name='loss_step1_weighted' )
            #LOG.debug("Adding loss. weight="+str(cclp_weight_per_chain))
            tf.losses.add_loss( loss_step_weighted )
    
    # --------------- Compute loss over markov chains of multiple length (steps) ----------------------
    M_class_match = tf.matmul(Phi_lu, Phi_lu, transpose_b=True) # [NxN] mtx. Dot prod, 0-1, confidence that two samples are of same class.
    H_Masked = H_trans * M_class_match # Eq.8. M can be seen as a soft mask applied to H, allowing steps only between same-class samples.
    
    for step_i in range( 2, cclp_steps+1 ):
        H_step = tf.matmul( H_Masked, H_step, transpose_b=False ) # Eq.8
        
        loss_step = tf.reduce_mean( - tf.reduce_sum( T_trans * tf.log( H_step + eps_log), axis=[1] ) )
        tf.summary.scalar('Loss_step'+str(step_i), loss_step)
        if cclp_steps==step_i or sum_over_chains :
            loss_cclp += loss_step
            if cclp_weight_per_chain > 0 :
                loss_step_weighted = tf.multiply(loss_step, cclp_weight_per_chain, name='loss_step'+str(step_i)+'_weighted')
                #LOG.debug("Adding loss for step_i="+str(step_i)+". weight="+str(cclp_weight_per_chain))
                tf.losses.add_loss( loss_step_weighted )
            
    tf.summary.scalar('Loss_cclp', loss_cclp)


# ***** Main function for setting up CCLP *****
# Note: Dont call this with an embedding Z_l/Z_u taken right at the output of a dropout layer.
#       Dropout makes the affinity matrix too sparse. Apply dropout right after, or at least a layer earlier.
def add_cclp_loss(Z_l,
                  Z_u,
                  y_l_lbls,
                  c_classes,
                  # Params for creating the graph
                  sim_metric = "dot",
                  l2_sigmas = 1.0,
                  l2_sigmas_trainable = False,
                  # Params for CCLP loss
                  cclp_weight = 1.0,
                  cclp_steps = 3,
                  sum_over_chains = True,
                  # Others
                  e_smooth = 0.00001,
                  optim_smooth_mtx = True ):
    # Z_l: matrix shape [ n_l, num_features_of_z_embed ]. Embeddings of labelled samples in batch.
    # Z_u: matrix shape [ n_u, num_features_of_z_embed ]. Embeddings of unlabelled samples in batch.
    # ... where n_l / n_u are number of labelled / unlabelled samples in a batch, respectively.
    # y_l_lbls: vector  [ n_l ]. It holds scalars, labels of samples in y_l. Not one hot.
    # c_classes: scalar. Number of classes in the task.
    # sim_metric: 'dot' or 'L2'. Similarity metrix to use for creating the graph and its affinity matrix.
    # l2_sigmas: std for scaling L2 distance. Can be int, or list of ints (one per feature of Z embedding).
    # l2_sigmas_trainable: Whether to learn sigmas during training.
    # cclp_weight: Float. The weight w in: Loss_total = Cross_Entr + w * Loss_cclp
    # cclp_steps: Number of steps of the longest markov chain along which to optimize the path between samples.
    # sum_over_chains: True/False. False: only the longest chain is optimized. True: sums over chains of diff lengths.
    # e_smooth: Epsilon for smoothing the transition matrix, to ensure it's invertible. CCLP insensitive to this.
    # optim_smooth_mtx: Whether CCLP loss should optimize actual or smoothed transition mtx H. CCLP insensitive to this.
    
    # Add summaries of configuration variables given for experiment.
    tf.summary.scalar('cclp_weight', cclp_weight)
    tf.summary.scalar('Z_cclp_steps', cclp_steps)
    # Helper. False if no unlabelled data used. E.g. to regularise fully supervised models.
    use_unlab = Z_u.get_shape()[0] > 1
    
    # ---------- Construct a graph over embeddings z(x) of all samples in batch. -----------
    # Compute transition matrix H of the graph.
    # Eq. 2,3,4 of CCLP paper.
    Z_lu = tf.concat( values=[ Z_l, Z_u ], axis=0 ) if use_unlab else Z_l
    H_trans = make_graph_and_transition_mtx( Z_lu, sim_metric, l2_sigmas, l2_sigmas_trainable )
    
    # ---------- Perform Label Propagation (LP) ---------------
    Y_l = tf.one_hot(y_l_lbls, c_classes) # Make training labels one hot matrix.
    # Smoothen transition mtx H. This ensures inverse mtx exists for LP, by making whole graph one connected component.
    # See Eq. 14 of Label Probagation, 2002 paper. Common practice before inverting transition matrices.
    n_lu = H_trans.get_shape().as_list()[0] # total number of samples in a batch ( n_l + n_u )
    uniform_p = 1.0 / float( n_lu ) # uniform probability 1/num-of-samples.
    H_trans_smooth = e_smooth * uniform_p + (1.0 - e_smooth) * H_trans
    
    if use_unlab: # If no unlabelled given (eg full supervision) dont do LP.
        Phi_u = label_propagation_to_unlabelled(Y_l, H_trans_smooth) # Returns class posteriors for unlabelled
    # Class posteriors after LP for all samples:
    Phi_lu = tf.concat( values=[ Y_l, Phi_u ], axis=0 ) if use_unlab else Y_l # [ N_lu x C_classes ] matrix
    
    # --------- Add CCLP cost to regularize feature space towards compact clustering -----------
    regularize_for_compact_clustering( H_trans = H_trans_smooth if optim_smooth_mtx else H_trans,
                                       Phi_lu = Phi_lu,
                                       cclp_weight = cclp_weight,
                                       cclp_steps = cclp_steps,
                                       sum_over_chains = sum_over_chains )
    
    
    
    
#####################################################################
#            Standard Cross Entropy - for labelled data             #
#####################################################################
    
def add_softmax_cross_entr(logits, lbls, weight=1.0, lbl_smoothing=0.0):
    # The standard supervised objective for labelled data.
    # labels: tensor of shape [batchsize], one scalar for each sample (number of true class).
    # weights: acts as a coefficient for the loss...
    # ... If a scalar is provided, then the loss is simply scaled by the given value.
    # ... If weights is a tensor of shape [batch_size], then the loss weights apply to each corresponding sample.
    
    # Internally, this calls tf.add_loss(), which adds loss to GraphKeys.LOSSES. ...
    # ... Those, along with regularization at layers, are all collected with a call tf.losses.get_total_loss().
    loss_logit_weighted = tf.losses.softmax_cross_entropy(
                            onehot_labels = tf.one_hot(lbls, logits.get_shape()[-1]),
                            logits = logits,
                            scope = 'loss_logit_weighted',
                            weights = weight,
                            label_smoothing = lbl_smoothing)
    tf.summary.scalar('Loss_Logit_weighted', loss_logit_weighted)
    
    
    
