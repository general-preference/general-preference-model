# export CUDA_VISIBLE_DEVICES=0

deepspeed train_rm_general_preference.py \
     --save_path ../results/saved_model/2b_gemma/rm \
     --save_steps -1 \
     --logging_steps 1 \
     --eval_steps 10 \
     --accumulated_gradient 1 \
     --micro_train_batch_size 2 \
     --pretrain google/gemma-2b \
     --bf16 \
     --max_epochs 3 \
     --max_len 2048 \
     --zero_stage 3 \
     --learning_rate 5e-7 \
     --general_preference_tau 0.1 \
     --dataset ../data/test_data/test_data.jsonl \
     --dataset_probs 1 \
     --flash_attn \
     --gradient_checkpointing \
     --group_size 1 \
     --value_head_dim 2 \
     --save_best_model 2 \
     --add_pretrain_loss \
     --ptx_loss_coef 0.01 \
     --is_general_preference \
     --train_split_ratio 0.98 \
     --save_best_model 2 


