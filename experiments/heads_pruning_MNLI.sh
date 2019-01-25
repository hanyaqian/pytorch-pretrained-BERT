#!/usr/bin/env bash

OPTIONS=${1:-""}

here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $here/prepare_task.sh MNLI

echo $base_acc
echo $part
for percent in `seq 10 10 100`
do
    echo -n "$percent	"
    prune_options="--do_prune --eval_pruned --prune_percent $percent $OPTIONS"
    acc=$(run_eval "$prune_options" | grep eval_accuracy | rev | cut -d" " -f1 | rev)
    echo "$acc"
done

