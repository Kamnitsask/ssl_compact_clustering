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

import datetime

def datetime_str() :
    #datetime returns in the format: YYYY-MM-DD HH:MM:SS.millis but ':' is not supported for Windows' naming convention.
    datetime_str = str(datetime.datetime.now().strftime("%Y.%m.%d.-%H.%M.%S")) # .strftime("%Y-%m-%d-%H-%M-%S"))
    return datetime_str













