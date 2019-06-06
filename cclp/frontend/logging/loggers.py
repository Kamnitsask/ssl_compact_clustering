#!/usr/bin/env python

# Copyright (c) 2018, Konstantinos Kamnitsas
#
# This program is free software; you can redistribute and/or modify
# it under the terms of the Apache License, Version 2.0. See the 
# accompanying LICENSE file or read the terms at:
# http://www.apache.org/licenses/LICENSE-2.0

from __future__ import absolute_import, division, print_function

import os
import logging

def get_loglevel_from_cmd_arg(cmd_loglevel=None):
    # assuming loglevel is bound to the string value obtained from the
    # command line argument. Convert to upper case to allow the user to
    # specify --log=DEBUG or --log=debug
    if cmd_loglevel is None:
        numeric_level = logging.DEBUG
    elif cmd_loglevel == 'd':
        numeric_level = getattr(logging, 'DEBUG', None)
    elif cmd_loglevel == 'i':
        numeric_level = getattr(logging, 'INFO', None)
    else:
        raise ValueError('Given loglevel letter in command line is invalid: %s' % cmd_loglevel)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % cmd_loglevel)
    return numeric_level
    
    
def setup_main_logger(cmd_loglevel=None, logfile=None):
    # create logger
    logger = logging.getLogger('main')
    numeric_level = get_loglevel_from_cmd_arg(cmd_loglevel)
    # the logger's level is the "lowest" level it will ever print.
    # The handlers can raise it up. They can never bring it down though (more details).
    logger.setLevel( logging.DEBUG )
    
    # create CONSOLE handler and set level
    cons_handler = logging.StreamHandler()
    cons_handler.setLevel( numeric_level )
    cons_formatter = logging.Formatter('%(asctime)s [%(name)s]: %(levelname)s: %(message)s \r') # create formatter. The \r allows the txt to look fine on windows. Doesnt otherwise.
    cons_handler.setFormatter(cons_formatter) # add formatter to cons_handler
    logger.addHandler(cons_handler) # add cons_handler to logger
    
    if logfile is not None:
        if not os.path.exists(logfile):
            os.mkdir(logfile)
        # create FILE handler and set level
        file_handler = logging.FileHandler( os.path.join(logfile, "log_"+logger.name+".txt") )
        file_handler.setLevel( logging.DEBUG )
        file_formatter = logging.Formatter('%(asctime)s [%(name)s]: %(levelname)s: %(message)s \r') # create formatter
        file_handler.setFormatter(file_formatter) # add formatter to file_handler
        logger.addHandler(file_handler) # add file_handler to logger
    
    
def setup_loggers(cmd_loglevel=None, logfile=None):
    # logfile = abs path to logfile.
    # This will create any loggers I wish, eg in case I want separate ones, for different classes
    # Eg I could have a "main" one, that is clean-ish, and one that keeps info for sampling? That is busier.
    # HOW-TO on loggers: https://docs.python.org/2/howto/logging.html
    logging.raiseExceptions = True # If True, when logging breaks, exception is printed. Process still continues though.
    setup_main_logger(cmd_loglevel, logfile)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    