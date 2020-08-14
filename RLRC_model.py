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
import configparser
from transformers import (BertTokenizer, PretrainedModel, DistilBertModel)

additional_special_tokens = ['[E1]', '[/E1]', '[E2]', '[/E2]']


class RC_CNN(nn.Module):
    def __init__(self, config):
        super(RC_CNN, self).__init__()
        self.config = config
        self.Bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # Test which is prefered, sequeantial classifier or cnn

        self. classifier = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 1, kernel_size=[3, 60], stride=[1, 60])),
            ('pool', nn.MaxPool2d(kernel_size=[70, 1], stride=1)),
            ('drop', nn.Dropout(p=config.keep_prob)),
            ('fc1', nn.Linear(self.config.cnn_size, self.config.num_classes))
        ]))
        '''
        self.conv1 = nn.Conv2d(1, 1, kernel_size=[3, 60], stride=[1, 60])
        self.pool = nn.MaxPool2d(kernel_size=[70, 1], stride=1)
        self.fc1 = nn.Linear(self.config.cnn_size, self.config.num_classes)
        self.drop = nn.Dropout(p=config.keep_prob)
        '''

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]

        logits = self.classifier(last_hidden_state_cls)
        return logits

    def save(self, save_dir='./models'):
        pass


if __name__ == "__main__":

    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased', do_lower_case=True, additional_special_tokens=additional_special_tokens)
