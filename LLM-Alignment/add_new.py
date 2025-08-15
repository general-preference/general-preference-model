name_pattern = "data-llama-3-8b-instruct-sppo-score-iter{}_gp_8b-table-0.002"
iters = [1, 2, 3]
base_model = "data-llama-3-8b-instruct-sppo-iter1"
remove_old = False

add_iter1_data = True
add_iter1_ranking = False

ckpt_dir = "checkpoints"
generated_dir = "generated"
ranking_dir = "ranking"

import os
import shutil


def handle_old(name_pattern, remove_old):
    for iter in iters:
        # 1. handle generated
        generated_path = os.path.join(generated_dir, name_pattern.format(iter))
        if os.path.exists(generated_path):
            if remove_old:
                shutil.rmtree(generated_path)
            else:
                generated_path_old = generated_path + "_old"
                if os.path.exists(generated_path_old):
                    shutil.rmtree(generated_path_old)
                os.rename(generated_path, generated_path_old)

        # 2. handle ranking
        ranking_path = os.path.join(ranking_dir, name_pattern.format(iter))
        if os.path.exists(ranking_path):
            if remove_old:
                shutil.rmtree(ranking_path)
            else:
                ranking_path_old = ranking_path + "_old"
                if os.path.exists(ranking_path_old):
                    shutil.rmtree(ranking_path_old)
                os.rename(ranking_path, ranking_path_old)


def add_new(name_pattern, base_model, add_iter1_data, add_iter1_ranking): 
    iter = 1
    name = name_pattern.format(iter)
    if not os.path.exists(os.path.join(generated_dir, base_model)):
        print("Base model does not exist")
        return
    
    # 1. if add iter1 data, copy all from base model dir
    if add_iter1_data:
        shutil.copytree(os.path.join(generated_dir, base_model), os.path.join(generated_dir, name))
    
    # 2. if add iter1 ranking, copy all from base model dir
    if add_iter1_ranking:
        shutil.copytree(os.path.join(ranking_dir, base_model), os.path.join(ranking_dir, name))


if __name__ == "__main__":
    handle_old(name_pattern, remove_old)
    add_new(name_pattern, base_model, add_iter1_data, add_iter1_ranking)
    print("Done")