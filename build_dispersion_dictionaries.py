import torch
import os
import operator
import math

def build_dis_dict(model, model_type):
    # model = 'seg'
    dis_path =  os.path.expanduser(f"~/Dir/projects/IPLVE/data/sorted_dispersion_{model}.txt")
    dis_sorted = open(dis_path).readlines()
    dis_sorted_info = [i.split(': ', 1)[0] for i in dis_sorted]

    dict_path = './data/dictionaries_id2w'
    dict_list = os.listdir(dict_path)
    seeds = [203, 255, 633, 813, 881]
    for seed in seeds:
        for dico in dict_list:
            if "eval" in dico and str(seed) in dico:
                dictionaries = open(os.path.join(dict_path, dico)).readlines()
                eval_ids_origin = [i.split(' ', 1)[0] for i in dictionaries]
                eval_words = [i.strip().split(' ', 1)[1] for i in dictionaries]
                if model_type == 'image':
                    sorted_index = [dis_sorted_info.index(i) for i in eval_ids_origin]
                else:
                    sorted_index = [dis_sorted_info.index(i) for i in eval_words]
                eval_zipped = list(zip(eval_ids_origin, eval_words, sorted_index))
                sorted_res = sorted(eval_zipped, key=operator.itemgetter(2))
                # eval_zipped.sort(key=dis_sorted_ids.index)
                block_size = math.ceil(len(sorted_res) / 3)
                id_final, word_final, _ = zip(*sorted_res)
                # id_final = list(id_final)
                # word_final = list(word_final)
                final_list = list(zip(id_final, word_final))
                # print(final_list[0:2])
                for block_idx, block_name in enumerate(['low', 'medium', 'high']):
                    with open(f'./data/dictionaries_{model_type}_dispersion/eval_{model}_{seed}_{block_name}.txt', 'w') as wd:
                        for (id_s, word) in final_list[block_idx*block_size : block_size * (block_idx + 1)]:
                            wd.write(f'{id_s} {word}\n')
                        wd.close()
                        # wd.write('\n'.join(sorted_res))

if __name__ == "__main__":
    model = 'bert'
    model_type = 'language'
    build_dis_dict(model, model_type)
                


