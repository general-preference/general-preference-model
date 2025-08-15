#!/bin/bash
export HF_HOME="/cephfs/shared/hf_cache"
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"


# iter_num=3
SUFFIX="_8B"
start_iter=3
iter_num=3
for i in $(seq 1 $iter_num); do
    if [ "$i" -eq 1 ]; then
        MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
    else
        MODEL=$OUTPUT_DIR
    fi
    OUTPUT_DIR="checkpoints/Llama-3-8B-Instruct-SPPO-Iter${i}${SUFFIX}-table"
    PROMPT="UCLA-AGI/data-mistral-7b-instruct-sppo-iter${i}"

    OUT="data-llama-3-8b-instruct-sppo-iter${i}-table${SUFFIX}" 
    DATASET="synthetic_data_llama-3-8b-instruct-sppo-iter${i}-table${SUFFIX}_score"

    if [ "$i" -lt $start_iter ]; then
        continue
    fi
    
    bash scripts/generate_table_8B.sh --model $MODEL --prompt $PROMPT --out_path $OUT
    bash scripts/pipeline_table.sh --model $MODEL --iter $i --dataset $DATASET --output_dir $OUTPUT_DIR --num 1
done