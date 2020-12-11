#!/bin/bash

# run.sh

# !! By default, runs on a single batch
# Set `QUICKRUN = False` in each of the files to run on the whole dataset

# --
# Huggingface

# Predict
python hf_main.py > hf_out.jl

# Score
python bleu.py hf_out.jl

# --
# Original

# Predict
python tf_main.py > tf_out.jl

# Score
python bleu.py tf_out.jl