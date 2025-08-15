from datasets import load_dataset
import json
import pandas as pd
import argparse
import os
import numpy as np
from transformers import AutoTokenizer

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2"
    )
    parser.add_argument('--output_dir', type=str, default='generated/iter1')
    parser.add_argument("--numgpu", type=int, default=8)
    parser.add_argument('--prompts', type=str, default='UCLA-AGI/data-mistral-7b-instruct-sppo-iter1')
    parser.add_argument('--data_frac', type=int, default=0)
    parser.add_argument('--frac_len', type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)  # local rank
    parser.add_argument("--pairs", type=int, default=5)
    return parser.parse_args()

def ranking(args, prompts, candidates):
    from RMs.reward_model import PairPreferenceLlama
    from RMs.ranker import Ranker

    model_name = "RLHFlow/pair-preference-model-LLaMA3-8B"
    rm = PairPreferenceLlama(model_name, f"cuda")
    ranker = Ranker(rm)

    print(f"Ranking {len(prompts)} prompts with {len(candidates[0])} candidates each")
    ranks, raw_table = ranker.rank(prompts, candidates, batch_size=1, return_raw_table=True)

    np.save(f"ranking/{args.output_dir}/{args.gpu}_{args.data_frac}.npy", ranks)
    np.save(f"ranking/{args.output_dir}/{args.gpu}_{args.data_frac}_table.npy", raw_table)


def split_prompts(prompts, frac_len, data_frac):
    if frac_len > 0:
        split_len = frac_len
        if split_len * (data_frac + 1) > len(prompts):
            return prompts[split_len * data_frac:]
        else:
            return prompts[split_len * data_frac: split_len * (data_frac + 1)]
    else:
        return prompts[:]


def apply_template(text, tokenizer):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}, {"role": "assistant", "content": "None"}],
        tokenize=False, add_generate_prompt=True
    ).split("None")[0]


def main(args):
    data = load_dataset(args.prompts, split="train")

    if "mistral" in args.model.lower():
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    elif "llama-3" in args.model.lower():
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    elif "gemma-2" in args.model.lower():
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    else:
        raise ValueError("Must contain model name in the dataset name. Supported models: Mistral/Llama-3")

    tokenizer.pad_token = tokenizer.eos_token

    prompts_all = [d["prompt"] for d in data]
    print(prompts_all[0])
    pairs = args.pairs
    all_generated = []

    for i in range(pairs):
        file_path = f"generated/{args.output_dir}/responses_{i}.json"
        with open(file_path) as f:
            gen = json.load(f)
            all_generated.append(gen)

    candidates_texts = list(zip(*all_generated))

    assert len(data) == len(candidates_texts)
    # NOTE
    # data = data[:len(candidates_texts)]
    print(f'Length of data: {len(data)}')

    data_frac = args.data_frac
    os.makedirs(f"ranking/{args.output_dir}", exist_ok=True)

    data_frac, frac_len = args.data_frac, args.frac_len
    prompts_all = split_prompts(prompts_all, frac_len, data_frac)
    candidates_texts = split_prompts(candidates_texts, frac_len, data_frac)

    ranking(args, prompts_all, candidates_texts)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
