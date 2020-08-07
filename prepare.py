# !/usr/bin/env python
# --------------------------------------------------------------
# File:          prepare.py
# Project:       RLRC
# Created:       Sunday, 5th July 2020 9:27:23 am
# @Author:       Molin Liu, MSc in Data Science, University of Glasgow
# Contact:       molin@live.cn
# Last Modified: Thursday, 23rd July 2020 3:45:12 pm
# Copyright  © Rockface 2019 - 2020
# --------------------------------------------------------------

import numpy as np
import pandas as pd
import os
from tqdm import tqdm


def _read_word_embedding():
    path_vec_file = './origin_data/vec.txt'
    with open(path_vec_file, encoding='utf-8') as vec_file:
        info = vec_file.readline()
        print(info)


def _add_entity_tag(row):
    token_sen = row['sen'].split()
    update_list_e1 = []
    update_list_e2 = []
    for i, j in enumerate(token_sen):
        if j == row['e1']:
            tmp = i+len(update_list_e1)+len(update_list_e2)
            update_list_e1.append(tmp)
            update_list_e1.append(tmp+2)
        if j == row['e2']:
            tmp = i+len(update_list_e1)+len(update_list_e2)
            update_list_e2.append(tmp)
            update_list_e2.append(tmp+2)

    for i, j in enumerate(update_list_e1):
        if i % 2 == 0:
            token_sen.insert(j, '[E1]')
        else:
            token_sen.insert(j, '[/E1]')

    for i, j in enumerate(update_list_e2):
        if i % 2 == 0:
            token_sen.insert(j, '[E2]')
        else:
            token_sen.insert(j, '[/E2]')

    return ' '.join(token_sen)


def prepare_bert_data(dataPath):
    full_data = pd.read_csv(dataPath, header=None, sep='\t').iloc[:, 2:]
    full_data.columns = ['e1', 'e2', 'rel', 'sen']
    tagged_sen = []
    with tqdm(total=len(full_data)) as pbar:
        for _, row in full_data.iterrows():
            tagged_sen.append(_add_entity_tag(row))
            pbar.update(1)
    #full_data['tagged_sen'] = full_data.apply(_add_entity_tag, axis=1)


def _clean_text(dataPath):
    output = []
    with open(dataPath, 'r') as origin_file:
        baselen = 0
        n_line = 1

        for line in origin_file.readlines():
            line = line.strip()
            token = line.split('\t')
            if baselen == 0:
                baselen = len(token)
            else:
                if len(token) != baselen:
                    print(token)
                    print(n_line)
            n_line += 1
            temp = '\t'.join(token[:6])+'\n'
            output.append(temp)
    os.rename(dataPath, dataPath[:-4]+'_original.txt')
    with open(dataPath, 'w') as outfile:
        outfile.writelines(output)


if __name__ == '__main__':
    prepare_bert_data(
        '/Users/meow/Documents/UofG/Disseration/RLRC/origin_data/test.txt')
