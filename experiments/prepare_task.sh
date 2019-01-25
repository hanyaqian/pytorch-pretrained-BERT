#!/usr/bin/env bash

TASK=$1
TRAIN_OPTIONS="${@:2}"

mkdir -p models
model_dir=models/$TASK
mkdir -p $model_dir

function run_train () {
    python examples/run_classifier.py $TRAIN_OPTIONS \
    --task_name $TASK \
    --do_train \
    --do_lower_case \
    --data_dir glue_data/$TASK/ \
    --bert_model bert-base-uncased \
    --max_seq_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir $model_dir 2>&1
}

function run_eval () {
    python examples/run_classifier.py \
    --task_name $TASK \
    --do_eval \
    --do_lower_case \
    $1 \
    --data_dir glue_data/$TASK/ \
    --bert_model bert-base-uncased \
    --max_seq_length 128 \
    --eval_batch_size 32 \
    --output_dir $model_dir 2>&1
}

if [ ! -f $model_dir/pytorch_model.bin ]
then
    run_train
fi

base_acc=$(run_eval "" | grep eval_accuracy | rev | cut -d" " -f1 | rev)