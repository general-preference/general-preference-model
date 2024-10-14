export CUDA_VISIBLE_DEVICES=0
python run_rm_rewardbench.py \
--model /cephfs/zhangge/root/code/sqz_general_preference/results/saved_model/gemma-2-2b-it/rm/batch32_tau01_no_sft_1e5_sky80k_cleaned_epoch2_vh8_w_moe_w_l2 \
--chat_template raw \
--bf16 \
--flash_attn \
--is_custom_model \
--do_not_save \
--model_name "test" \
--batch_size 64 \
--value_head_dim 8 \
--max_length 4096 \
--is_general_preference \
--add_prompt_head 

