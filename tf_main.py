#!/usr/bin/env python

"""
    tf_translate.py
    
    tensorflow==2.2
    tensorflow-hub==0.8.0
    tensorflow-text==2.2.1
"""

import sys
import json
import numpy as np
from tqdm import tqdm

import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import tensorflow_text as tf_text

import datasets

tf.disable_eager_execution()

QUICKRUN = True

# --
# Load dataset

dataset = datasets.load_dataset("wmt14", "de-en", split="test")
dataset = list(dataset)
dataset = [xx['translation'] for xx in dataset]
dataset = [dataset[i] for i in np.random.RandomState(123).permutation(len(dataset))]

# --
# Load model

model = hub.Module('https://tfhub.dev/google/bertseq2seq/bert24_de_en/1')

# --
# Setup session

sess = tf.InteractiveSession()
sess.run(tf.tables_initializer())
sess.run(tf.global_variables_initializer())

src       = tf.placeholder(tf.string, shape=[None])
translate = model(src)

# --
# Translate

for chunk in tqdm(np.array_split(dataset, len(dataset) // 32)):
    
    inputs  = np.array([xx['de'] for xx in chunk])
    targets = np.array([xx['en'] for xx in chunk])
    
    output_str = sess.run(translate, feed_dict = {
        src : inputs
    })
    
    for output, target in zip(output_str, targets):
        print(json.dumps({
            "output" : output.decode('utf-8'),
            "target" : str(target),
        }))
        sys.stdout.flush()
    
    if QUICKRUN:
        break
