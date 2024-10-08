# Adapted from https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/cli/train_rm.py

import argparse
import math
import os
from datetime import datetime
from transformers.trainer import get_scheduler
from general_preference.datasets import GeneralRewardDataset
from general_preference.models import get_reward_model
from general_preference.trainer import GeneralPreferenceRewardTrainer
from general_preference.utils import blending_datasets, get_strategy, get_tokenizer

def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    # load huggingface model/config
    model = get_reward_model(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(),
        init_value_head=True,
        is_general_preference=args.is_general_preference,
        value_head_dim=args.value_head_dim,
        init_prompt_head=True,
        add_prompt_head=args.add_prompt_head
    )

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model, "left", strategy, use_fast=not args.disable_fast_tokenizer)
    tokenizer.truncation_side = "right"
    
    strategy.print(model)

    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.l2)

    # prepare for data and dataset
    total_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=5000000,
        stopping_strategy="all_exhausted"
    )
    if args.train_split_ratio < 1:
        total_data_lengh = min(args.max_samples, len(total_data))
        train_data = total_data.select(range(int(total_data_lengh * args.train_split_ratio)))
        eval_data = total_data.select(range(int(total_data_lengh * args.train_split_ratio), total_data_lengh))
        
        train_dataset = GeneralRewardDataset(train_data, tokenizer, args.max_len, strategy, is_custom=args.is_custom_dataset, return_prompt_length=args.return_prompt_length)
        train_dataloader = strategy.setup_dataloader(
            train_dataset,
            args.micro_train_batch_size,
            True,
            True,
            train_dataset.collate_fn,
            group_size=args.group_size, 
        )
        if len(eval_data) != 0:
            eval_dataset = GeneralRewardDataset(eval_data, tokenizer, args.max_len, strategy, is_custom=args.is_custom_dataset, return_prompt_length=args.return_prompt_length)
            eval_dataloader = strategy.setup_dataloader(
                eval_dataset, args.micro_train_batch_size, True, False, eval_dataset.collate_fn
            )
        else:
            eval_dataloader=None
            strategy.print("No separate validation data split was detected. The entire dataset will be utilized for training.")
    else:
        train_dataset = GeneralRewardDataset(total_data, tokenizer, args.max_len, strategy, is_custom=args.is_custom_dataset, return_prompt_length=args.return_prompt_length)
        train_dataloader = strategy.setup_dataloader(
            train_dataset,
            args.micro_train_batch_size,
            True,
            True,
            train_dataset.collate_fn,
            group_size=args.group_size, 
        )
        eval_dataloader=None
        strategy.print("No separate validation data split was detected. The entire dataset will be utilized for training.")
        
    # scheduler
    num_update_steps_per_epoch = len(train_dataloader) // strategy.accumulated_gradient
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        "cosine", #### "cosine"  "constant"
        optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps,
    )

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # strategy prepare
    (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))

    if args.load_checkpoint:
        strategy.print("Load checkpoint: ", args.save_path)

    os.makedirs(args.save_path, exist_ok=True)

    # batch_size here is micro_batch_size * 2
    # we use merged chosen + rejected response forward
    trainer = GeneralPreferenceRewardTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_epochs=args.max_epochs,
        is_general_preference=args.is_general_preference,
        tau=args.general_preference_tau,
        value_head_dim=args.value_head_dim,
    )

    trainer.fit(args)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, tokenizer, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default="google/gemma-2b")
    parser.add_argument("--dataset", type=str, default="../data/test_data/test_data.jsonl")
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--save_path", type=str, default="../results/saved_model")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="../results/saved_model/checkpoint")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1000)  # 1000GB
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--micro_train_batch_size", type=int, default=8)
    parser.add_argument("--accumulated_gradient", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--compute_fp32_loss", action="store_true", default=False)
    parser.add_argument("--margin_loss", action="store_true", default=False)
    parser.add_argument("--grad_accum_dtype", type=str, default=None)
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--input_template", type=str, default="Human:\n{}\nAssistant:\n")
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    
    # custom arguments added 
    parser.add_argument("--is_custom_dataset", action="store_true", default=False, help="Whether to use custom cyclic dataset. Default to False.")
    parser.add_argument("--is_general_preference", action="store_true", default=False, help="Whether to use General Preference model. Default to False (Bradley Terry model by default).")
    parser.add_argument("--general_preference_tau", type=float, default=0.1, help="Hyperparameter tau used in general preference loss.")
    parser.add_argument("--group_size", type=int, default=1, help="Number of data to group together during shuffling.")
    parser.add_argument("--value_head_dim", type=int, default=2, help="Dimension of the value head in the general preference model. Ignored by the Bradley Terry model. Should be even.")
    parser.add_argument("--save_best_model", type=int, default=None, help="Save the top N models with the lowest evaluation loss.")
    parser.add_argument("--add_pretrain_loss", action="store_true", default=False, help="Include the pretraining loss of chosen inputs in the total loss calculation.")
    parser.add_argument("--ptx_loss_coef", type=float, default=0.1, help="coefficient for pretraining loss included in the total loss.")
    parser.add_argument("--train_split_ratio", type=float, default=1, help="Ratio of the dataset to use for training. (1-train_split_ratio) for validation. Should not exceed 1.")
    parser.add_argument("--reward_scaler_beta", type=float, default=2.0, help="A constant that controls the scaling of the reward difference.")
    parser.add_argument("--reward_margin", type=float, default=1.0, help="Chosen response exceeds rejected reward by at least reward_margin. A hyperparameter for DPORefFree Loss.")
    parser.add_argument("--regression_target_margin", type=float, default=10.0, help="Target regression margin. A hyperparameter for Regression Loss.")
    parser.add_argument("--return_prompt_length", action="store_true", default=False, help="Return the prompt length in the dataset collator if set. Default to False. Should set to be True when 'add_prompt_head' is True.")
    parser.add_argument("--add_prompt_head", action="store_true", default=False, help="Add a prompt head to the model if set. Default to False.")
    
    # wandb pamameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="train_rm_general_preference")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="rm_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    args = parser.parse_args()
    train(args)
