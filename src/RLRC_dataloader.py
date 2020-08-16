import pandas as pd
import numpy as np
import pickle
import torch
import os
from tqdm import tqdm
from transformers import (BertTokenizer, DistilBertModel, DistilBertTokenizer)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split

def input_data(origin_data = './origin_data/train_filtered.txt'):
    train_data = pd.read_csv(
        origin_data, 
        sep='\t')
    return train_data

def get_entity_mask(input_ids, e1_id_start, e1_id_end, e2_id_start, e2_id_end):
    if input_ids.shape != torch.Size([1, 128]):
        print(input_ids)
    assert input_ids.shape == torch.Size([1, 128])
    entity1_mask = np.zeros([1, len(input_ids[0])])
    entity2_mask = np.zeros([1, len(input_ids[0])])
    
    #input_ids = input_ids.numpy()
    #e1_start = np.where(input_ids==e1_id_start)
    try:
        e1_start = (input_ids==e1_id_start).nonzero()[0][1].item()
        e1_end = (input_ids==e1_id_end).nonzero()[0][1].item()
        e2_start = (input_ids==e2_id_start).nonzero()[0][1].item()
        e2_end = (input_ids==e2_id_end).nonzero()[0][1].item()
        entity1_mask[0][e1_start+1:e1_end]=1
        entity2_mask[0][e2_start+1:e2_end]=1
        #print(entity1_mask)
        #print(entity2_mask)
        
        return torch.tensor(entity1_mask), torch.tensor(entity2_mask)
    except:
        print(input_ids)
        exit()


def get_bert_tokenizer(pretrain_model="distilbert-base-uncased"):
    tokenizer = DistilBertTokenizer.from_pretrained(
        pretrain_model, do_lower_case=False)
    tokenizer.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]'])
    return tokenizer


def generate_BertData(data_dir = './data'):
    print('Create BertData from scratch.')
    
    pretrain_model = "distilbert-base-uncased"
    tokenizer = get_bert_tokenizer(pretrain_model)

    if data_dir == './tests':
        train_data = input_data('./tests/data/train.txt')
    else:
        train_data = input_data()
    max_len = 128
    input_ids = []
    attention_masks = []
    e1_masks = []
    e2_masks = []
    labels = []
    num = 0
    e1_id_start = tokenizer.convert_tokens_to_ids('[E1]')
    e1_id_end = tokenizer.convert_tokens_to_ids('[/E1]')
    e2_id_start = tokenizer.convert_tokens_to_ids('[E2]')
    e2_id_end = tokenizer.convert_tokens_to_ids('[/E2]')

    for sent in tqdm(train_data['sen']):
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
        e1, e2 = get_entity_mask(encoded_dict['input_ids'], e1_id_start, e1_id_end, e2_id_start, e2_id_end)
        e1_masks.append(e1)
        e2_masks.append(e2)
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
    train_labels = torch.tensor(train_data['rel_id'], dtype=torch.long)
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    e1_masks = torch.cat(e1_masks, dim=0)
    e2_masks = torch.cat(e2_masks, dim=0)

    '''
    Dump result to .pkl files
    '''
    with open(os.path.join(data_dir, 'bert_labels.pkl'), 'wb') as f:
        pickle.dump(train_labels, f)
    with open(os.path.join(data_dir, 'bert_input_ids.pkl'), 'wb') as f:
        pickle.dump(input_ids, f)
    with open(os.path.join(data_dir, 'bert_attention_masks.pkl'), 'wb') as f:
        pickle.dump(attention_masks, f)
    with open(os.path.join(data_dir, 'e1_mask.pkl'), 'wb') as f:
        pickle.dump(e1_masks, f)
    with open(os.path.join(data_dir, 'e2_mask.pkl'), 'wb') as f:
        pickle.dump(e2_masks, f)
    return [
        os.path.join(data_dir, 'bert_labels.pkl'), 
        os.path.join(data_dir, 'bert_input_ids.pkl'),
        os.path.join(data_dir, 'bert_attention_masks.pkl'),
        os.path.join(data_dir, 'e1_mask.pkl'),
        os.path.join(data_dir, 'e2_mask.pkl'),
        ]

def get_BertData(data_dir = './data'):
    if not os.path.exists(os.path.join(data_dir, "bert_labels.pkl")):
        generate_BertData(data_dir)
    print('Load Bert data from .pkl files...')
    with open(os.path.join(data_dir, "bert_labels.pkl"), 'rb') as f:
        train_labels = pickle.load(f)
    with open(os.path.join(data_dir, 'bert_input_ids.pkl'), 'rb') as f:
        input_ids = pickle.load(f)
    with open(os.path.join(data_dir, 'bert_attention_masks.pkl'), 'rb') as f:
        attention_masks = pickle.load(f)
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
    get_BertData()
