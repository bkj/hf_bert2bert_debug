#!/usr/bin/env python

"""
    rouge.py
"""

import sys
import json
import sacrebleu

# --
# Load predictions

f = sys.argv[1].strip()
x = [json.loads(xx) for xx in open(f).readlines()]

print(x)
print(f'n_records={len(x)}', file=sys.stderr)

output = [xx['output'] for xx in x]
target = [xx['target'] for xx in x]

# --
# Compute metrics

metrics = sacrebleu.corpus_bleu(output, [target], lowercase=False)
print('w/o lowercase', metrics.score)

metrics = sacrebleu.corpus_bleu(output, [target], lowercase=True)
print('w/  lowercase', metrics.score)