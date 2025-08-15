#!/bin/bash
export HF_HOME="/cephfs/shared/hf_cache"
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"


# iter_num=3
if [ -z $beta ]; then
    beta=0.001
fi
if [ -z $lr ]; then
    lr=5e-7
fi
if [ -z $clamp_thres ]; then
    clamp_thres=1000
fi

echo "beta: $beta"
echo "lr: $lr"
echo "clamp_thres: $clamp_thres"

export BETA=$beta
export CLAMP_THRES=$clamp_thres

# RM_MODEL_NAME="/cephfs/shared/zhangge/models/general_preference/model_revise/gemma-2b-it/batch32_tau01_no_sft_1e5_sky80k_cleaned_epoch2_vh8_w_moe_w_l2"
RM_MODEL_NAME="/cephfs/shared/zhangge/models/general_preference/model_revise/Llama-31-8b-Instruct/batch32_tau01_no_sft_2e6_sky80k_cleaned_epoch2_vh4_w_moe_w_l2"

# set RM_CONFIGS as a json string
# RM_CONFIGS="{\"is_general_preference\": true, \"tau\": 0.1, \"value_head_dim\": 8, \"add_prompt_head\": true}"
RM_CONFIGS="{\"is_general_preference\": true, \"tau\": 0.1, \"value_head_dim\": 4, \"add_prompt_head\": true}"

# RM_MODEL_SUFFIX="gp_2b"
RM_MODEL_SUFFIX="gp_8b"

LR_SUFFIX=""
if [ "$lr" != "5e-7" ]; then
    LR_SUFFIX="_${lr}"
fi

SUFFIX="_${RM_MODEL_SUFFIX}${LR_SUFFIX}"
export RM_MODEL_NAME    
export SUFFIX
export RM_CONFIGS

start_iter=1
iter_num=3
for i in $(seq 1 $iter_num); do
    echo "iter $i"
    if [ "$i" -eq 1 ]; then
        MODEL=""google/gemma-2-9b-it""
    else
        MODEL=$OUTPUT_DIR
    fi
    OUTPUT_DIR="checkpoints/Gemma-2-9B-It-SPPO-score-Iter${i}${SUFFIX}-table"
    PROMPT="UCLA-AGI/data-mistral-7b-instruct-sppo-iter${i}"

    OUT="data-gemma-2-9b-it-sppo-score-iter${i}-table${SUFFIX}"
    DATASET="synthetic_data_gemma-2-9b-it-sppo-score-iter${i}-table${SUFFIX}_score"

    if [ "$i" -lt $start_iter ]; then
        continue
    fi
    
    bash scripts/generate_score_table_gp.sh --model $MODEL --prompt $PROMPT --out_path $OUT
    bash scripts/pipeline_score_table.sh --model $MODEL --iter $i --dataset $DATASET --output_dir $OUTPUT_DIR --num 1 --beta $beta --learning_rate $lr --batch_size 2
done