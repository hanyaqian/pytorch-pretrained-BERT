#!/usr/bin/env bash

OPTIONS=${1:-""}

here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $here/prepare_task.sh MRPC

echo $base_acc
echo $part
echo "Layer \\textbackslash~Head & 1 & 2 & 3 & 4 \\\\"
for layer in `seq 1 12`
do
    echo -n "$layer"
    for head in `seq 1 12`
    do
        mask_str="${layer}:${head}"
        acc=$(run_eval "--attention-mask-heads $mask_str $OPTIONS" | grep eval_accuracy | rev | cut -d" " -f1 | rev)
        printf " & %.5f" $(echo "$acc - $base_acc" | bc )
    done
    echo " \\\\"
done

