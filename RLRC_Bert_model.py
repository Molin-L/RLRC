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
from collections import OrderedDict
from transformers import (BertTokenizer, DistilBertModel, DistilBertTokenizer)

additional_special_tokens = ['[E1]', '[/E1]', '[E2]', '[/E2]']


class RC_BERT(nn.Module):
    def __init__(self, config):
        super(RC_BERT, self).__init__()
        cnn_size = 230

        self.config = config
        self.Bert = DistilBertModel.from_pretrained(
            'distilbert-base-uncased',
            num_labels=self.config['num_classes'],
            output_attentions=False,
            output_hidden_states=False
        )
        # Test which is prefered, sequeantial classifier or cnn
        '''
        self.classifier = nn.Sequential(OrderedDict([
            
            ('conv1', nn.Conv2d(1, 1, kernel_size=[3, 60], stride=[1, 60])),
            ('pool', nn.MaxPool2d(kernel_size=[70, 1], stride=1)),
            ('drop', nn.Dropout(p=self.config['dropout'])),
            ('fc1', nn.Linear(cnn_size, self.config['num_classes']))
        ]))
        '''
        self.classifier = nn.Sequential(nn.Linear(768, 250),
                                        nn.ReLU(),
                                        # nn.Dropout(0.5),
                                        nn.Linear(250, 53))
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            self.config['pretrain_model'], do_lower_case=False)
        self.tokenizer.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]'])
        e1_id = self.tokenizer.convert_tokens_to_ids('[E1]')
        e2_id = self.tokenizer.convert_tokens_to_ids('[E2]')
        assert e1_id != e2_id != 1
        self.Bert.resize_token_embeddings(len(self.tokenizer))
        self.Bert.cuda()
        '''
        self.conv1 = nn.Conv2d(1, 1, kernel_size=[3, 60], stride=[1, 60])
        self.pool = nn.MaxPool2d(kernel_size=[70, 1], stride=1)
        self.fc1 = nn.Linear(self.config.cnn_size, self.config.num_classes)
        self.drop = nn.Dropout(p=config.keep_prob)
        '''

    def forward(self, input_ids, attention_mask):
        outputs = self.Bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]

        logits = self.classifier(last_hidden_state_cls)

        '''
        @Todo: Add entity mask
        '''
        return logits

    def save(self, save_dir='./models'):
        pass


if __name__ == "__main__":
    config = {
        'pretrain_model': "distilbert-base-uncased",
        'num_classes': 53,
        'lr': 0.001,
        'dropout': 0.5,
        'epochs': 3
    }
    RC_BERT(config)
