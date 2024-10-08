export CUDA_VISIBLE_DEVICES=0
python run_rm_rewardbench.py \
--model general-preference/GPM-Llama-3.1-8B \
--chat_template raw \
--bf16 \
--flash_attn \
--is_custom_model \
--do_not_save \
--model_name "general-preference/GPM-Llama-3.1-8B" \
--batch_size 64 \
--value_head_dim 6 \
--max_length 4096 \
--is_general_preference \
--add_prompt_head 

