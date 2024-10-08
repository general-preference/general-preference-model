
python batch_inference_rm_general_preference.py \
--pretrain ../results/saved_model/2b_gemma_it/rm/batch32_tau01_no_sft_1e5_sky80k_epoch1_vh8_no_moe_no_l2 \
--dataset  ../data/test_data/test_data.jsonl  \
--max_samples 100000 \
--general_preference_tau 0.1 \
--micro_batch_size 3 \
--max_len 2048 \
--value_head_dim 2 \
--is_custom_dataset \
--is_general_preference \


