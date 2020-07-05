# !/usr/bin/env python
# --------------------------------------------------------------
# File:          logger.py
# Project:       RLRC
# Created:       Sunday, 5th July 2020 7:03:14 pm
# @Author:       Molin Liu, MSc in Data Science, University of Glasgow
# Contact:       molin@live.cn
# Last Modified: Sunday, 5th July 2020 7:03:25 pm
# Copyright  © Rockface 2019 - 2020
# --------------------------------------------------------------
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)
