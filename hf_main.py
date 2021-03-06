#!/usr/bin/env python

"""
    hf_main.py
"""

import os
import sys
import json
import numpy as np
from tqdm import tqdm

import torch

import datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

QUICKRUN = True

# --
# Load dataset

dataset = datasets.load_dataset("wmt14", "de-en", split="test")
dataset = list(dataset)
dataset = [xx['translation'] for xx in dataset]
dataset = [dataset[i] for i in np.random.RandomState(123).permutation(len(dataset))]

# --
# Load model

tokenizer = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_de_en", pad_token="<pad>", eos_token="</s>", bos_token="<s>")
model     = AutoModelForSeq2SeqLM.from_pretrained("google/bert2bert_L-24_wmt_de_en").cuda()

# --
# Dataloader

def collate_fn(batch):
    inputs  = [xx['de'] for xx in batch]
    inputs  = tokenizer(inputs, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    
    targets = [xx['en'] for xx in batch]
    
    return inputs, targets


dataloader = torch.utils.data.DataLoader(
    dataset    = dataset,
    batch_size = 32,
    collate_fn = collate_fn,
)

# --
# Run prediction

for inputs, targets in tqdm(dataloader):
    
    input_ids      = inputs.input_ids.cuda()
    attention_mask = inputs.attention_mask.cuda()
    
    output_ids     = model.generate(input_ids, attention_mask=attention_mask)
    output_str     = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    
    for output, target in zip(output_str, targets):
        print(json.dumps({
            "output" : output,
            "target" : target,
        }))
        
        sys.stdout.flush()
        
    if QUICKRUN:
        break
