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
from transformers import (BertTokenizer, DistilBertModel, DistilBertTokenizer, BertPreTrainedModel, BertModel, BertForSequenceClassification)
from torch.nn import MSELoss, CrossEntropyLoss
additional_special_tokens = ['<e1>', '</e1>', '<e2>', '</e2>']

def l2_loss(parameters):
    return torch.sum(
        torch.tensor([
            torch.sum(p ** 2) / 2 for p in parameters if p.requires_grad
        ]))


class RC_BERT(nn.Module):
    def __init__(self, config):
        super(RC_BERT, self).__init__()
        cnn_size = 230

        self.config = config
        self.l2_reg_lambda = self.config['l2_reg_lambda']
        self.num_labels = self.config['num_classes']
        if self.config['pretrain_model']=='distilbert-base-uncased':
            self.Bert = DistilBertModel.from_pretrained(
                'distilbert-base-uncased',
                num_labels=self.config['num_classes'],
                output_attentions=False,
                output_hidden_states=False
            )
        else:
            self.Bert = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased", 
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
        self.classifier = nn.Sequential(nn.Linear(2304, 250),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(250, 53))
        self.dropout = nn.Dropout(0.5)
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            self.config['pretrain_model'], do_lower_case=False)
        self.tokenizer.add_tokens(['<e1>', '</e1>', '<e2>', '</e2>'])
        e1_id = self.tokenizer.convert_tokens_to_ids('<e1>')
        e2_id = self.tokenizer.convert_tokens_to_ids('<e2>')
        assert e1_id != e2_id != 1
        self.Bert.resize_token_embeddings(len(self.tokenizer))
        self.Bert.cuda()
        '''
        self.conv1 = nn.Conv2d(1, 1, kernel_size=[3, 60], stride=[1, 60])
        self.pool = nn.MaxPool2d(kernel_size=[70, 1], stride=1)
        self.fc1 = nn.Linear(self.config.cnn_size, self.config.num_classes)
        self.drop = nn.Dropout(p=config.keep_prob)
        '''

    def forward(self, input_ids, attention_mask=None, e1_mask=None, e2_mask=None, labels=None):
        outputs = self.Bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0][:, :, :]
        pooled_output = outputs[0][:, 0, :]
        #sequence_output = outputs[0][0]
        
        def extract_entity(sequence_output, e_mask):
            extended_e_mask = e_mask.unsqueeze(1)
            extended_e_mask = torch.bmm(
                extended_e_mask.float(), sequence_output).squeeze(1)
            return extended_e_mask.float()
        e1_h = extract_entity(sequence_output, e1_mask)
        e2_h = extract_entity(sequence_output, e2_mask)
        context = self.dropout(pooled_output)
        pooled_output = torch.cat([context, e1_h, e2_h], dim=-1)
        print(pooled_output.size())
        
        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]
        
        device = logits.get_device()
        l2 = l2_loss(self.parameters())
        # 
        if device >= 0:
            l2 = l2.to(device)
        loss = l2 * self.l2_reg_lambda
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss += loss_fct(logits.view(-1), labels.view(-1))
            else:
                # loss_fct = CrossEntropyLoss()
                # loss += loss_fct(
                #     logits.view(-1, self.num_labels), labels.view(-1))
                # I thought that using Gumbel softmax should be better than the following code.

                probabilities = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
                one_hot_labels = F.one_hot(labels, num_classes=self.num_labels)
                if device >= 0:
                    one_hot_labels = one_hot_labels.to(device)

                dist = one_hot_labels[:, 1:].float() * log_probs[:, 1:]
                example_loss_except_other, _ = dist.min(dim=-1)
                per_example_loss = - example_loss_except_other.mean()

                rc_probabilities = probabilities - probabilities * one_hot_labels.float()
                second_pre,  _ = rc_probabilities[:, 1:].max(dim=-1)
                rc_loss = - (1 - second_pre).log().mean()

                #
                loss += per_example_loss + 5 * rc_loss

            outputs = (loss,) + outputs
        
        return outputs
    def save(self, save_dir='./out'):
        with open(os.path.join(save_dir, 'model.pth'), 'wb') as f:
            torch.save(self.state_dict(), f)
        

class Interaction():

    def __init__(self, config):
        self.model = config['model']
    
    def reward(self, batch_input_ids, batch_attention_masks, batch_train_labels, batch_e1_masks, batch_e2_masks):
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
