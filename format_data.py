# !/usr/bin/env python
# --------------------------------------------------------------
# File:          format_data.py
# Project:       RLRC
# Created:       Sunday, 5th July 2020 9:31:26 am
# @Author:       Molin Liu, MSc in Data Science, University of Glasgow
# Contact:       molin@live.cn
# Last Modified: Sunday, 5th July 2020 9:31:28 am
# Copyright  © Rockface 2019 - 2020
# --------------------------------------------------------------

import json
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from tqdm import tqdm
import re
import unidecode
import logging
from logger import logger
logger = logging.getLogger(__name__)

# Set validation data ratio
valid_ratio = 0.1


def _line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)


def _clean(entity):
    entity = entity.lower()
    result = re.sub('[\W_]+', '_', entity)
    result = unidecode.unidecode(result)
    return result


def _valid_set(train_file):
    valid_set = []
    train_set = []
    valid_file = os.path.join(os.path.dirname(train_file), 'valid.json')
    if os.path.exists(valid_file):
        exit()
    with open(train_file) as json_file:
        json_reader = json.load(json_file)
        i = 0
        for sen in json_reader:
            if i < len(json_reader):
                valid_set.append(sen)
            else:
                train_set.append(sen)
    with open(valid_file, 'w') as out_json:
        json.dump(valid_set, out_json, indent=4, sort_keys=True)
    logger.info("Write to %s" % valid_file)
    with open(train_file, 'w') as out_json:
        json.dump(train_set, out_json, indent=4, sort_keys=True)
    logger.info("Write to %s" % train_file)


def _clean_entity(file):
    content = []
    char_pattern = re.compile('')
    with open(file, 'r+') as f:
        entity_df = pd.read_csv(
            f, sep="\t", index_col=None, header=None, skiprows=1)
        entity_df.iloc[:][0] = entity_df.iloc[:][0].apply(_clean)
        entity_df.to_csv(file, header=False, index=False, sep='\t')
        _line_prepender(file, str(len(entity_df)))


def _fetch_data_nyt10(inPath):
    if not os.path.exists(inPath):
        logger.error("%s not found" % inPath)
        logger.error(os.listdir(os.path.dirname(inPath)))
        raise FileNotFoundError
    with open(inPath) as json_file:
        json_reader = json.load(json_file)

        e1_list = []
        e2_list = []
        rel_list = []
        sentence_list = []

        logger.info("Read from json file...")
        for sen in tqdm(json_reader):
            e1_list.append(sen['head']['word'])
            e2_list.append(sen['tail']['word'])
            rel_list.append(sen['relation'])
            sentence_list.append("%s" % sen['sentence'])

        outfile = inPath[:-4]+'csv'
        logger.info("Preparing data for %s..." % outfile)
        data_dict = {'Entity1': e1_list, 'Entity2': e2_list,
                     'Relation': rel_list, 'Sentence': sentence_list}
        out_frame = pd.DataFrame.from_dict(data_dict)
        out_frame.to_csv(outfile, index=False)
        logger.info("Finish write %s" % outfile)

        return e1_list, e2_list, rel_list


def preprocess_nyt10(inPath):
    test_path = os.path.join(inPath, 'test.json')
    train_path = os.path.join(inPath, 'train.json')
    le_entity = preprocessing.LabelEncoder()
    le_rel = preprocessing.LabelEncoder()

    train_e1, train_e2, train_rel = _fetch_data_nyt10(train_path)
    test_e1, test_e2, test_rel = _fetch_data_nyt10(test_path)
    """
    Generate test and train data for word embedding
    For more detail, check: https://github.com/thunlp/OpenKE#data
    """
    """
    Convert entities to id
    """
    entity_list = train_e1 + train_e2 + test_e1 + test_e2
    le_entity.fit(entity_list)
    entities_list = le_entity.classes_
    entities_df = pd.DataFrame(
        {'entitiy': entities_list, 'id': np.arange(len(entities_list))})
    entities_out_file = os.path.join(inPath, 'entity2id.csv')
    entities_df.to_csv(entities_out_file, header=False, index=False, sep='\t')
    _line_prepender(entities_out_file, str(len(entities_df)))
    logger.info("Finish write %s" % entities_out_file)

    train_e1_id = le_entity.transform(train_e1)
    train_e2_id = le_entity.transform(train_e2)

    test_e1_id = le_entity.transform(test_e1)
    test_e2_id = le_entity.transform(test_e2)
    """
    Convert relations to id
    """
    le_rel.fit(train_rel+test_rel)
    train_rel_id = le_rel.transform(train_rel)
    test_rel_id = le_rel.transform(test_rel)

    rel_map_list = le_rel.classes_
    rel_df = pd.DataFrame(
        {'relation': rel_map_list, 'id': np.arange(len(rel_map_list))})
    rel_out_file = os.path.join(inPath, 'relation2id.csv')
    rel_df.to_csv(rel_out_file, header=False, index=False, sep='\t')
    _line_prepender(rel_out_file, str(len(rel_df)))
    logger.info("Finish write %s" % rel_out_file)

    train_dataid_dict = {'entity1': train_e1_id,
                         'entity2': train_e2_id, 'relation': train_rel_id}
    train_id_df = pd.DataFrame.from_dict(train_dataid_dict)
    train_id_file = os.path.join(inPath, 'train2id.csv')
    train_id_df.iloc[int(valid_ratio*len(train_id_df)):][:].to_csv(
        train_id_file, index=False, header=False, sep='\t')
    _line_prepender(train_id_file, str(
        len(train_id_df.iloc[int(valid_ratio*len(train_id_df)):][:])))
    valid_id_file = os.path.join(inPath, 'valid2id.csv')
    train_id_df.iloc[:int(valid_ratio*len(train_id_df))
                     ][:].to_csv(valid_id_file, index=False, header=False, sep='\t')
    _line_prepender(valid_id_file, str(
        len(train_id_df.iloc[:int(valid_ratio*len(train_id_df))][:])))
    logger.info("Finish write %s" % train_id_file)

    test_dataid_dict = {'entity1': test_e1_id,
                        'entity2': test_e2_id, 'relation': test_rel_id}

    test_id_df = pd.DataFrame.from_dict(test_dataid_dict)
    test_id_file = os.path.join(inPath, 'test2id.csv')
    test_id_df.to_csv(test_id_file, index=False, header=False, sep='\t')
    _line_prepender(test_id_file, str(len(test_id_df)))
    logger.info("Finish write %s" % test_id_file)


if __name__ == "__main__":
    preprocess_nyt10('data/NYT10')
    _clean_entity(
        '/Users/meow/Documents/Projects/UoG_Proj/RLRC/data/shrink/entity2id.txt')
    _valid_set(
        "/Users/meow/Documents/Projects/UoG_Proj/RLRC/data/shrink/train.json")
