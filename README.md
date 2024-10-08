# General Preference Model (GPM)

General Preference Modeling with Preference Representations for Aligning Language Models, [https://arxiv.org/abs/2410.02197](https://arxiv.org/abs/2410.02197)

As Huggingface Daily Papers: [https://huggingface.co/papers/2410.02197](https://huggingface.co/papers/2410.02197)

## Introduction

This repository is designed for training and evaluating the General Preference representation model (GPM). It includes the following:

* Training code for both GPM and BT reward models.

* Evaluation code adapted from [RewardBench](https://github.com/allenai/reward-bench) for evaluating GPM and BT RM.

## Key Components:

`scripts/run_train_rm_general_preference_single.sh`: Run training for reward models on a single node. `scripts/run_train_rm_general_preference_multi.sh`: Run training for reward models across multiple nodes.  
`rewardbench_eval/run_rm_rewardbench.sh`: Run RewardBench evaluations for reward models.
`eval/batch_inference_rm_general_preference.sh`: Run evaluations on a custom-defined dataset.
`general_preference`: Useful code for training, heavily based on [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF).

## Quick Start
### Installation
rewardbench==0.1.2 depends on transformers==4.44.0. Install rewardbench first to avoid dependency conflicts. 
```bash
git clone https://github.com/general-preference/general-preference-model
cd general-preference-model
pip install rewardbench==0.1.2
pip install -e .
```
Reinstall deepspeed with specific build options. 
```bash
export DS_BUILD_SPARSE_ATTN=0; export DS_BUILD_EVOFORMER_ATTN=0; DS_BUILD_OPS=1 pip install deepspeed==0.13.5 --no-cache --force-reinstall
```
If the installation fails, install torch before other packages.
```bash
pip install torch==2.3.0
pip install -e .
```

## BibTeX
