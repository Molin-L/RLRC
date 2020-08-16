import pytest
import os
import sys
sys.path.append(".")
import src
#from src import RLRC_dataloader
from src.RLRC_dataloader import (
    get_bert_tokenizer, input_data, get_entity_mask, get_BertData, generate_BertData)
import logging
import pandas as pd
from tqdm import tqdm

@pytest.fixture(scope='module')
def set_tokenizer():
    return get_bert_tokenizer()

def clean_test_files(func):
    def inner(*args, **kwargs):
        ret = func(*args, **kwargs)
        print(ret)
        for file in ret:
            os.remove(file)
        return ret
    return inner
    
def test_load_data():
    data = input_data()
    print(data.describe())
    print(data.info())


def test_tokenizer_entity(set_tokenizer):
    tokenizer = set_tokenizer
    e1_id = tokenizer.convert_tokens_to_ids('[E1]')
    e2_id = tokenizer.convert_tokens_to_ids('[E2]')
    assert e1_id != e2_id != 1


def test_tokenizer_encode_plus(set_tokenizer):
    tokenizer = set_tokenizer
    sent = "[CLS] sen. charles e. schumer called on federal safety officials yesterday to reopen their investigation into the fatal crash of a passenger jet in [E2] belle_harbor [/E2] , [E1] queens [/E1] , because equipment failure , not pilot error , might have been the cause . [SEP]"
    encoded_dict = tokenizer.encode_plus(
        sent,
        add_special_tokens=False,
        max_length=128,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True,
        return_special_tokens_mask=True
    )
    e1_id = tokenizer.convert_tokens_to_ids('[E1]')
    e2_id = tokenizer.convert_tokens_to_ids('[E2]')
    print(e1_id)
    print(encoded_dict.input_ids)
    print(encoded_dict.special_tokens_mask)

def test_get_entity_mask():
    tokenizer = get_bert_tokenizer()
    sent = "[CLS] sen. charles e. schumer called on federal safety officials yesterday to reopen their investigation into the fatal crash of a passenger jet in [E2] belle_harbor [/E2] , [E1] queens [/E1] , because equipment failure , not pilot error , might have been the cause . [SEP]"
    encoded_dict = tokenizer.encode_plus(
        sent,
        add_special_tokens=False,
        max_length=128,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True,
        return_special_tokens_mask=True
    )
    e1_id_start = tokenizer.convert_tokens_to_ids('[E1]')
    e1_id_end = tokenizer.convert_tokens_to_ids('[/E1]')
    e2_id_start = tokenizer.convert_tokens_to_ids('[E2]')
    e2_id_end = tokenizer.convert_tokens_to_ids('[/E2]')
    e1_mask, e2_mask = get_entity_mask(encoded_dict.input_ids, e1_id_start, e1_id_end, e2_id_start, e2_id_end)
    assert e1_mask.shape == e2_mask.shape == encoded_dict.input_ids.shape


def test_verify_dataset(set_tokenizer):
    toknenizer = set_tokenizer
    train_data = input_data()
    pretrain_model = "distilbert-base-uncased"
    tokenizer = get_bert_tokenizer(pretrain_model)

    max_len = 128
    num = 0
    e1_id_start = tokenizer.convert_tokens_to_ids('[E1]')
    e1_id_end = tokenizer.convert_tokens_to_ids('[/E1]')
    e2_id_start = tokenizer.convert_tokens_to_ids('[E2]')
    e2_id_end = tokenizer.convert_tokens_to_ids('[/E2]')

    for sent in train_data['sen']:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=False,
            max_length=max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        input_ids = encoded_dict['input_ids']
        if not ((e1_id_start in input_ids) and (e2_id_start in input_ids)):
            print(sent)
        assert (e1_id_start in input_ids) and (e2_id_start in input_ids)

@clean_test_files
def test_generate_BertData():
    return generate_BertData('./tests')

