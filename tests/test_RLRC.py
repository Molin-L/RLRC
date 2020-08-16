
from src.RLRC_dataloader import get_bert_tokenizer, input_data
import pytest
import pandas as pd
from logger import logger


def test_load_data():
    data = input_data()
    logger.info(data.iloc[0, 1])


def test_tokenizer_entity():
    tokenizer = get_bert_tokenizer()
    e1_id = tokenizer.convert_tokens_to_ids('[E1]')
    e2_id = tokenizer.convert_tokens_to_ids('[E2]')
    assert e1_id != e2_id != 1


def test_tokenizer_encode_plus():
    tokenizer = get_bert_tokenizer()
    sent = "[CLS] sen. charles e. schumer called on federal safety officials yesterday to reopen their investigation into the fatal crash of a passenger jet in [E2] belle_harbor [/E2] , [E1] queens [/E1] , because equipment failure , not pilot error , might have been the cause . [SEP]"
    encoded_dict = tokenizer.encode_plus(
        sent,
        add_special_tokens=False,
        max_length=128,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )


test_tokenizer_encode_plus()
