
deepspeed --force_multi --hostfile hostfile train_rm_general_preference.py \
     --save_path ../results/saved_model/2b_gemma/rm \
     --save_steps -1 \
     --logging_steps 1 \
     --eval_steps 1000000 \
     --accumulated_gradient 1 \
     --micro_train_batch_size 16 \
     --pretrain google/gemma-2b \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 3 \
     --learning_rate 2e-6 \
     --general_preference_tau 1 \
     --dataset ../data/test_data/test_data.jsonl \
     --dataset_probs 1 \
     --flash_attn \
     --gradient_checkpointing \
     --group_size 1 \
     --value_head_dim 2 \
     --save_best_model 2 \
     --ptx_loss_coef 0.1 \
     --train_split_ratio 1 \
     --save_best_model 2 \
     --is_general_preference \
     --use_wandb True 

     # --is_general_preference \
     # --add_pretrain_loss \
     # --is_custom_dataset \
     # --return_prompt_length \
     # --add_prompt_head \






