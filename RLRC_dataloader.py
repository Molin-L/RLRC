import pandas as pd
import numpy as np
import pickle
import torch
import os
from tqdm import tqdm
from transformers import (BertTokenizer, DistilBertModel, DistilBertTokenizer)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split


def input_data():
    train_data = pd.read_csv(
        './origin_data/sentence_128_retag.txt', sep='\t', header=None)
    return train_data


def get_BertData():
    if os.path.exists("./data/bert_labels.pkl"):
        print('Load Bert data from .pkl files...')
        with open('./data/bert_labels.pkl', 'rb') as f:
            # torch.tensor(train_data.iloc[:, 0])
            train_labels = pickle.load(f)
        with open('./data/bert_input_ids.pkl', 'rb') as f:
            input_ids = pickle.load(f)  # torch.cat(input_ids, dim=0)
        with open('./data/bert_attention_masks.pkl', 'rb') as f:
            # torch.cat(attention_masks, dim=0)
            attention_masks = pickle.load(f)
        return input_ids, attention_masks, train_labels
    train_data = input_data()
    pretrain_model = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(
        pretrain_model, do_lower_case=False)
    print(len(tokenizer))
    tokenizer.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]'])
    print(len(tokenizer))
    e1_id = tokenizer.convert_tokens_to_ids('[E1]')
    e2_id = tokenizer.convert_tokens_to_ids('[E2]')

    print(e1_id, e2_id)
    assert e1_id != e2_id != 1

    max_len = 128
    input_ids = []
    attention_masks = []
    labels = []
    num = 0
    for sent in tqdm(train_data.iloc[:, 1]):
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=False,
            max_length=max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
    train_labels = torch.tensor(train_data.iloc[:, 0])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    with open('./data/bert_labels.pkl', 'wb') as f:
        pickle.dump(train_labels, f)

    with open('./data/bert_input_ids.pkl', 'wb') as f:
        pickle.dump(input_ids, f)

    with open('./data/bert_attention_masks.pkl', 'wb') as f:
        pickle.dump(attention_masks, f)

    return input_ids, attention_masks, train_labels


def create_dataset():
    # Combine the training inputs into a TensorDataset.
    input_ids, attention_masks, train_labels = get_BertData()
    dataset = TensorDataset(input_ids, attention_masks, train_labels)

    # Create a 90-10 train-validation split.

    # Calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print('\t{:>5,} training samples'.format(train_size))
    print('\t{:>5,} validation samples'.format(val_size))

    return train_dataset, val_dataset


def get_dataloader(batch_size=32):
    train_dataset, val_dataset = create_dataset()

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        # Pull out batches sequentially.
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size  # Evaluate with this batch size.
    )
    return train_dataloader, validation_dataloader


if __name__ == "__main__":
    get_dataloader()
