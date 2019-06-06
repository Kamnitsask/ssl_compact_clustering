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


def logistic_growth_model(progress_step, total_growth_steps, final_val, init_val, steepness_param=5.):
    # Growing a variable from zero to final_val value in a sigmoid fashion.
    # progress_step: Current step of model training. ie the tf.get_global_step(). No delay. step>0. Ie, growth starts at 0. Delay should have been taken care earlier. 
    # total_growth_steps: Total number of steps for the sigmoid to saturate at the target. Note that at half the steps it reaches exactly half the final value.
    # final_val: Final value. Must be > 0 positive.
    # Returns: A TF tensor value that should have the value of the logistic growth model at x=progress_step.
    
    # I think this version of mine should work for both positive and negative final values. Though I must debug properly.
    progress_step = tf.cast(progress_step, tf.float32)
    total_growth_steps = tf.cast(total_growth_steps, tf.float32)
    # it's like a sigmoid, from 0 to final_val in 'total_growth_steps'. In steps/2, it reaches exactly final_val/2)
    steepness = steepness_param / total_growth_steps
    tanh_steep_centered_on_totalD2_steps = tf.tanh( steepness * (progress_step - total_growth_steps / 2. ) )
    
    log_model_y = (tanh_steep_centered_on_totalD2_steps + 1.) / 2.
    return init_val + (final_val-init_val) * log_model_y
    
def linear_model(progress_step, total_growth_steps, final_val, init_val):
    # progress_step: >0. Growth starts at 0 step, there is no delay.
    # ... Delay should have been taken care of earlier. ie, if there is delay, pass here progress_step = step-delay.
    # Value can both increase and decrease
    if total_growth_steps == 0.0:
        return final_val
    else :
        return progress_step * ( final_val - init_val ) / (1.0*total_growth_steps) + init_val

# This follows very closely LBA, to reproduce their reported results closely.
def apply_growth(type, step, value_init, value_fin, delay, growth_length):
    # step: current iteration.
    # value_init: init value, before growth.
    # value_fin: final value, after end of growth.
    # delay: number of steps to wait before growth.
    # growth_length: number of steps to grow.
    assert growth_length > 0
    progress_step = tf.cast(step - delay, tf.float32)
    
    if type is None:
        value = value_fin
    elif step <= delay:
        value = value_init
    elif step >= delay + growth_length:
        value = value_fin
    elif type in ['sigmoid']:
        value = logistic_growth_model(progress_step, growth_length, value_fin)
    elif type in ['linear', 'lin']:
        value = linear_model(progress_step, growth_length, value_fin)
    else:
        raise NameError('Invalid type: ' + str(type))
    
    val = tf.clip_by_value(value, 0., value_fin) # Ensure it's not negative nor over final value.
    return val


