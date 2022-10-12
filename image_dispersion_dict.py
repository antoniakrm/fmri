import torch
import os
import math
from build_dico_shuffle import dico_build

def split_dict(sim_path, dict_path, block_num=5, seed=-1):

    name_sim = open(sim_path).readlines()
    id_name_pairs = open(dict_path).readlines()
    ids = [i.split(': ')[0] for i in id_name_pairs]
    name_with_ids = [i.strip().split(': ')[1] for i in id_name_pairs]
    name_with_sim = [i.split(': ')[0] for i in name_sim]
    sorted_ids = [ids[name_with_ids.index(i)] for i in name_with_sim]

    # sorted_ids = list(dict.fromkeys(sorted_ids))
    # cos_sim = [i.strip().split(': ')[1] for i in name_sim]

    dico = dico_build(sorted_ids, id_list=sorted_ids, name_list=name_with_sim, seed=seed)

    total_nums = len(dico)
    block_size = math.ceil(total_nums / block_num)

    sorted_ids = [sorted_ids[i*block_size:i*block_size + block_size] for i in range(block_num)]

    return sorted_ids

def main():
    sim_path = './data/sorted_similarities_seg.txt'
    seed = -1
    categories_id_path = os.path.expanduser('~/Dir/projects/IPLVE/data/image_ids_wiki_using.txt')
    splitted_ids = split_dict(sim_path=sim_path, dict_path=categories_id_path,seed=seed)
    dico = 

if __name__ == "__main__":
    main()
