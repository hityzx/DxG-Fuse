#!/bin/bash

MODEL_DIR="./pretrained/bert-base-uncased"
VIT_DIR="./pretrained/vit"

TRAIN_JSON="./data/amg/train.json"
VAL_JSON="./data/amg/val.json"
TEST_JSON="./data/amg/test.json"


python train.py --train_json "$TRAIN_JSON" --val_json "$VAL_JSON" --test_json "$TEST_JSON" --bert_model "$MODEL_DIR" --vit_model "$VIT_DIR" --batch_size 30 --gpu_id 0 --patience 10 --max_seq_len 128 --dropout 0.1 --learning_rate 2e-5 --weight_decay 0.01 --seed 42
