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
from sklearn import preprocessing
from tqdm import tqdm
import logging
from logger import logger
logger = logging.getLogger(__name__)


def preprocess_nyt10(infile):
    with open(infile) as json_file:
        json_reader = json.load(json_file)
        le = preprocessing.LabelEncoder()
        e1_list = []
        e2_list = []
        rel_list = []
        sentence_list = []

        logger.info("Read from json file...")
        for p in tqdm(json_reader):
            e1_list.append(p['head']['word'])
            e2_list.append(p['tail']['word'])
            rel_list.append(p['relation'])
            sentence_list.append("%s" % p['sentence'])

        outfile = infile[:-4]+'csv'
        logger.info("Preparing data for %s..." % outfile)
        data_dict = {'Entity1': e1_list, 'Entity2': e2_list,
                     'Relation': rel_list, 'Sentence': sentence_list}
        out_frame = pd.DataFrame.from_dict(data_dict)
        out_frame.to_csv(outfile, index=False)
        logger.info("Finish write %s" % outfile)
        """
        Generate test and train data for word embedding
        """
        """
        Convert entities to id
        """

        entity_list = e1_list + e2_list
        le.fit(entity_list)
        entities_list = le.classes_
        entities_df = pd.DataFrame(entities_list)
        entities_out_file = infile[:-5]+'_entity2id.csv'
        entities_df.to_csv(entities_out_file, header=False)
        logger.info("Finish write %s" % entities_out_file)
        """
        Convert relations to id
        """
        e1id_list = le.transform(e1_list)
        e2id_list = le.transform(e2_list)
        relid_list = le.fit_transform(rel_list)
        rel_map_list = le.classes_
        rel_df = pd.DataFrame(rel_map_list)
        rel_out_file = infile[:-5]+'_rel2id.csv'
        rel_df.to_csv(rel_out_file, header=False)
        logger.info("Finish write %s" % rel_out_file)

        dataid_dict = {'entity1': e1id_list,
                       'entity2': e2id_list, 'relation': relid_list}
        outid_frame = pd.DataFrame.from_dict(dataid_dict)
        outid_file = infile[:-5]+'_id.csv'
        outid_frame.to_csv(outid_file, index=False, header=False)
        logger.info("Finish write %s" % outid_file)


if __name__ == "__main__":
    preprocess_nyt10('data/NYT10/test.json')
    preprocess_nyt10('data/NYT10/train.json')
