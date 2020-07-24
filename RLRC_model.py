# !/usr/bin/env python
# --------------------------------------------------------------
# File:          RLRC_model.py
# Project:       RLRC
# Created:       Sunday, 5th July 2020 5:43:14 pm
# @Author:       Molin Liu, MSc in Data Science, University of Glasgow
# Contact:       molin@live.cn
# Last Modified: Sunday, 5th July 2020 5:43:15 pm
# Copyright  © Rockface 2019 - 2020
# --------------------------------------------------------------
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class RC_CNN(nn.module):
    def __init__(self):
        pass

    def save(self, save_dir='./models'):
        pass


if __name__ == "__main__":
    with open("data/NYT10/RE/vec.bin", 'rb') as f:
        i = 0
        for line in f.readlines():
            print(line.decode("utf-8", "ignore"))
            i += 1
            if i > 10:
                break
