# !/usr/bin/env python
# --------------------------------------------------------------
# File:          preprocess.py
# Project:       RLRC
# Created:       Sunday, 9th August 2020 6:28:08 pm
# @Author:       Molin Liu, MSc in Data Science, University of Glasgow
# Contact:       molin@live.cn
# Last Modified: Sunday, 9th August 2020 6:28:12 pm
# Copyright  © Rockface 2019 - 2020
# --------------------------------------------------------------

import numpy as np
import pandas as pd
import copy
from RLRC_dataloader import get_bert_tokenizer
from sklearn import preprocessing
from tqdm import tqdm

def _add_entity_tag(row):
    token_sen = row['sen'].split()
    out_token_sen = copy.deepcopy(token_sen)
    update_list_e1 = []
    update_list_e2 = []
    has_e1 = False
    has_e2 = False
    for i, j in enumerate(token_sen):
        if j == row['e1']:
            has_e1=True
            tmp = i+len(update_list_e1)+len(update_list_e2)
            out_token_sen.insert(tmp, '[E1]')
            out_token_sen.insert(tmp+2, '[/E1]')
            
            update_list_e1.append(tmp)
            update_list_e1.append(tmp+2)
        if j == row['e2']:
            has_e2=True
            tmp = i+len(update_list_e1)+len(update_list_e2)
            update_list_e2.append(tmp)
            update_list_e2.append(tmp+2)
            out_token_sen.insert(tmp, '[E2]')
            out_token_sen.insert(tmp+2, '[/E2]')
    temp_row = copy.deepcopy(row)
    temp_row['sen'] = ' '.join(out_token_sen)
    if not (has_e1 and has_e2):
        return False, False
    return ' '.join(out_token_sen), temp_row

def prepare_bert_data(dataPath):
    max_len = 110
    full_data = pd.read_csv(dataPath, header=None, sep='\t').iloc[:, 2:]
    full_data.columns = ['e1', 'e2', 'rel', 'sen']
    tokenizer = get_bert_tokenizer()
    tagged_sen = []
    row_list = []
    with tqdm(total=len(full_data)) as pbar:
        for _, row in full_data.iterrows():
            temp_sen, temp_row = _add_entity_tag(row)
            if temp_sen:
                tagged_sen.append(temp_sen)
                if len(tokenizer.tokenize(temp_sen))<max_len:
                    row_list.append(temp_row)
            pbar.update(1)
    
    cleaned_df = pd.DataFrame(row_list)
    cleaned_df = cleaned_df.fillna(value='UNK')
    cleaned_df = cleaned_df.iloc[:, 2:]
    cleaned_df = add_label(cleaned_df)
    print(cleaned_df.describe())
    cleaned_df.to_csv(dataPath[:-4]+'_filtered.txt', index=False, sep='\t')

def add_label(input_df):
    #input_df = pd.read_csv('./origin_data/train_filtered.txt', sep='\t')
    le = preprocessing.LabelEncoder()
    label_df = pd.read_csv('./origin_data/relation2id.txt', sep=' ', header=None)
    trans_df = label_df.set_index(0).T
    label_dict = trans_df.to_dict('records')
    input_df['rel_id'] = input_df['rel'].replace(label_dict[0])
    input_df['rel_id'] = input_df['rel_id'].apply(pd.to_numeric())

    return input_df

prepare_bert_data('./origin_data/train.txt')