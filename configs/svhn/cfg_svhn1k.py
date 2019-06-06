#!/usr/bin/env python

# Variables needed for pre-setting up the session.
session_type = 'train'

# ========== Variables needed for the session itself. ========
# === Variables that are read from the cmd line too. ===
# WARN: Values given in cmd line overwrite these given below.
out_path = "./output/svhn1k/"
device = None # GPU device number
model_to_load = None # To start from pretrained model/continue training.
plot_save_emb = 0 # Plot embedding if emb_size is 2D. 0: No. 1: save. 2: plot. 3: save & plot.

# === Variables not given from command line ===
# --- Network architecture ---
model = {'feat_extractor_z': 'svhn_cnn', 'emb_z_size': 128, 'emb_z_act': 'elu',
         'batch_norm_decay': 0.99, 'l2_reg_feat': 1e-6, 'l2_reg_classif': 1e-6, 'classifier_dropout': 0.5}

# --- Validation, labelled and unlabelled folds for training ---
dataset = 'svhn'  # mnist, svhn, cifar
val_on_test = True  # If True, validate on test dataset. Else, validate on subset of training data.
num_val_samples = -1  # How many samples to use for validation when training. -1 to use all.
num_lbl_samples = 1000  # How many labelled data to learn from. -1 to use all.
num_unlbl_samples = -1  # How many unlabelled data to learn from. -1 to use all.
unlbl_overlap_val = True  # If True and val_on_test=False, unlabelled samples may overlap with validation.
unlbl_overlap_lbl = True  # If True, unlabelled samples can overlap with labelled. If False, they are disjoint.
# --- Batch sampling, normalization, augmentation ---
n_lbl_per_class_per_batch = 10 # How many labelled samples per class in a batch.
n_unlbl_per_batch = 100 # How many unlabelled samples in a batch.
norm_imgs_pre_aug = "zscoreDb" # None, zscoreDb, zscorePerCase, center0sc1, rescale01, zca, zca_center0sc1.
augm_params = {}
# Augmentation options:
# augm_params = { "reflect":{"apply": True},
#    "color_inv": {"apply": False, "params": {"p":0.5}},
#    "rand_crop": {"apply": False, "params": {'transf':[2,2]}},
#    "blur": {"apply": False, "params": {'p':0.5, 's':1.0, 'sc':1}} }
seed = None

# --- Training loop ---
max_iters = 75000 # Max training iterations
val_during_train = True # Whether to validate performance every now and then.
val_interval = 500 # Every how many training steps to validate performance.
# Learning rate schedule
lr_sched_type = 'piecewise' # 'expon_decay' or 'piecewise'
lr_expon_init = None  # Only for expon. Initial LR.
lr_expon_decay_factor = None  # Only for expon. How much to decrease.
lr_expon_decay_steps = None  # Only for expon. How often to decrease.
lr_piecewise_boundaries = [40000, 55000, 75000] # Only for expon. When to change LR.
lr_piecewise_values = [3e-4,   1e-4,  3e-5,   3e-5] # Only for expon. Initial and following values.

# --- Compact Clustering via Label Propagation (CCLP) ---
cc_weight = 1.0 # Weight w in: Ltotal = Lsup + w*Lcclp . Set to 0 to disable CCLP.
cc_steps = 3 # Length of longest chain to optimize. Set to 0 to disable CCLP.
cc_loss_on = (cc_steps > 0) or (cc_weight > 0) # Set to False to disable.
# Params for creating the graph.
cc_sim_metric = "dot" # dot or L2, similarity metric for creating the graph.
cc_l2_sigmas_init = 1.0 # Only for L2. float or list of floats per dim.
cc_l2_sigmas_trainable = True # Only for L2. Whether to learn the sigmas.
cc_l2_sigmas_lr_multipl = 1.0 # Only for L2.
# Secondary configs for CCLP.
cc_sum_over_chains = True # If False, only the longest chain is optimized.
cc_e_smooth = 0.00001
cc_optim_smooth_mtx = True

