import os
import random
import math
from itertools import product

def dico_build(ids_words_path, seed):
    if not os.path.exists(f'./data/dictionaries/image_classes_dict_wiki_seed_{seed}.txt'):
        pairs = open(ids_words_path).readlines()
        id_all = [i.strip().split(': ')[0] for i in pairs]
        name_all = [i.strip().lower().split(': ')[1] for i in pairs]
        
        related_dict = {}
        for id_v in id_all:
            related_names = []
            for idx, id_val in enumerate(id_all):
                if id_v == id_val:
                    name = name_all[idx]
                    related_names.append(name)
            related_dict[id_v] = related_names
        
        # keys = list(related_dict.keys())
        # random.seed(seed)
        # random.shuffle(keys)
        # shuffled_dict = dict()
        # for key in keys:
        #     shuffled_dict.update({key: related_dict[key]})

        dico = []
        # for value in shuffled_dict.values():
        for value in related_dict.values():
            relations = list(product(value, repeat=2))
            for src, tgt in relations:
                word_dico = f'{src}    {tgt}'
                dico.append(word_dico)

        new_dico = list(dict.fromkeys(dico))
        with open(f'./data/dictionaries/image_classes_dict_wiki_seed_{seed}.txt', 'w') as idname_w:
            idname_w.write('\n'.join(new_dico))
            idname_w.close()
    
    all_dico = open(f'./data/dictionaries/image_classes_dict_wiki_seed_{seed}.txt').readlines()
    all_dico_size = len(all_dico)

    train_nums = math.ceil(all_dico_size * 0.7)
    validation_nums = math.ceil(all_dico_size * 0.85)
    # test_nums = all_dico_size - validation_nums

    with open(f'./data/dictionaries/train_wiki_dico_{seed}.txt', 'w+') as biw:
        
        for iword in all_dico[:train_nums]:
            biw.write(iword)
        biw.close()

    with open(f'./data/dictionaries/eval_wiki_dico_{seed}.txt', 'w+') as biw:
        for iword in all_dico[train_nums:validation_nums]:
            biw.write(iword)
        biw.close()

    with open(f'./data/dictionaries/test_wiki_dico_{seed}.txt', 'w+') as biw:
        for iword in all_dico[validation_nums:]:
            biw.write(iword)
        biw.close()

if __name__ == '__main__':
    ids_words_path = './data/image_ids_wiki_using.txt'

    if not os.path.exists(f'./data/dictionaries'):
        os.mkdir('./data/dictionaries')

    # for _ in range(5):
    #     seed = random.randint(0, 100)
    #     dico_build(ids_words_path, seed)
    seed = 'no'
    dico_build(ids_words_path, seed)