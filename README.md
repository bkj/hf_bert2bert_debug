# hf_bert2bert_debug

Comparing results of WMT14 DE->EN evaluation on:
 - original tensorflow implementation: https://tfhub.dev/google/bertseq2seq/bert24_de_en/1
 - ported huggingface implementation: google/bert2bert_L-24_wmt_de_en

In our experiments, the ported huggingface implementation has substantially lower BLEU scores and lower quality translations.

See `./run.sh` for usage.
