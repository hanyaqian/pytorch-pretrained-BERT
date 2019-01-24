#!/usr/bin/env bash

OPTIONS=${1:-""}

mkdir -p models
model_dir=models/MNLI
mkdir -p $model_dir

function run_train () {
    python examples/run_classifier.py \
    --task_name MNLI \
    --do_train \
    --do_lower_case \
    --data_dir glue_data/MNLI/ \
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
    --task_name MNLI \
    --do_eval \
    --do_lower_case \
    --attention-mask-heads $1 $2 \
    --data_dir glue_data/MNLI/ \
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
echo $base_acc
echo $part
echo "Layer \\textbackslash~Head & 1 & 2 & 3 & 4 \\\\"
for layer in `seq 1 12`
do
    echo -n "$layer"
    for head in `seq 1 12`
    do
        mask_str="${layer}:${head}"
        acc=$(run_eval $mask_str "$OPTIONS" | grep eval_accuracy | rev | cut -d" " -f1 | rev)
        printf " & %.5f" $(echo "$acc - $base_acc" | bc )
    done
    echo " \\\\"
done

