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

## Example Usage of the GPM

Below is an example code snippet (see `./gpm_example_usage.py`):

```python
prompt_text = "Describe the importance of reading books in today's digital age."
response1 = "Books remain crucial in the digital era, offering in-depth knowledge and fostering critical thinking. They provide a unique, immersive experience that digital media can't replicate, contributing significantly to personal and intellectual growth."
response2 = "Books are still useful for learning new things. They help you relax and can be a good break from screens."

context1 = [
    {"role": "user", "content": prompt_text},
    {"role": "assistant", "content": response1}
]

context2 = [
    {"role": "user", "content": prompt_text},
    {"role": "assistant", "content": response2}
]

rm = GPMPipeline("general-preference/GPM-Llama-3.1-8B", value_head_dim=6)

reward1, prompt_hidden_state = rm([context1], return_prompt=True)
reward2 = rm([context2])

result = generate_high_dim_result_with_prompt(rm.model, rm.value_head_dim, reward1, reward2, prompt_hidden_state)

result_batch = result.float().cpu().detach().numpy().tolist()

results = []
[
    results.append(1) if result > 0 else results.append(0)
    for result in result_batch
]

print(result_batch)
```

## Citations

Please cite the paper and star this repo if you use the General Preference Representation Model (GPM) and General Preference Optimization (GPO) and find it interesting/useful, thanks! Feel free to contact zhangge19951114@gmail.com | zhangyif21@mails.tsinghua.edu.cn or open an issue if you have any questions.

```
@article{zhang2024general,
  title={General Preference Modeling with Preference Representations for Aligning Language Models},
  author={Zhang, Yifan and Zhang, Ge and Wu, Yue and Xu, Kangping and Gu, Quanquan},
  journal={arXiv preprint arXiv:2410.02197},
  year={2024}
}
```
