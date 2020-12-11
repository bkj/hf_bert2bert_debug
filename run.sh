#!/bin/bash

# run.sh

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