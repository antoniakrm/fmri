import os
import random
import math
import itertools

def dico_build(ids_words_path, seed, id_list=None, name_list=None):
    # if not os.path.exists(f'./data/dictionaries/image_classes_dict_wiki_seed_{seed}.txt'):
    pairs = open(ids_words_path).readlines()
    id_all = [i.strip().split(': ')[0] for i in pairs]
    name_all = [i.strip().lower().split(': ')[1] for i in pairs]

    if id_list:
        id_all = id_list
    if name_list:
        name_all = name_list
    
    related_dict = {}
    for id_v in id_all:
        related_names = []
        for idx, id_val in enumerate(id_all):
            if id_v == id_val:
                name = name_all[idx]
                related_names.append(name)
        related_dict[id_v] = related_names
    id_nums = len(related_dict)

    if seed > -1:
        keys = list(related_dict.keys())
        random.seed(seed)
        random.shuffle(keys)
        shuffled_dict = dict()
        for key in keys:
            shuffled_dict.update({key: related_dict[key]})
    else:
        shuffled_dict = related_dict
    
    i = iter(shuffled_dict.items())
    train_part = dict(itertools.islice(i, math.ceil(len(shuffled_dict) * 0.7)))
    eval_part = dict(itertools.islice(i, math.ceil(len(shuffled_dict) * 0.15)))
    test_part = dict(i)
    # build id2w dict
    train_dico = []
    eval_dico = []
    test_dico = []

    for key, values in train_part.items():
        for value in values:
            train_dico.append(f'{key} {value}\n')

    for key, values in eval_part.items():
        for value in values:
            eval_dico.append(f'{key} {value}\n')

    for key, values in test_part.items():
        for value in values:
            test_dico.append(f'{key} {value}\n')

    return train_dico, eval_dico, test_dico
    # build w2w dict
    # dico = []
    # for value in shuffled_dict.values():
    # # for value in related_dict.values():
    #     relations = list(product(value, repeat=2))
    #     for src, tgt in relations:
    #         word_dico = f'{src}    {tgt}'
    #         dico.append(word_dico)

    # new_dico = list(dict.fromkeys(dico))
    # return new_dico

def dico_write(write_dir, dicos, seed):
    for idx, part in enumerate(["train","eval","test"]):
        with open(f'{write_dir}/{part}_wiki_dico_{seed}.txt', 'w+') as biw:
            for iword in dicos[idx]:
                biw.write(iword)
            biw.close()
    # if not os.path.exists(f'{write_dir}/image_classes_dict_wiki_seed_{seed}.txt'):
    #     with open(f'{write_dir}/image_classes_dict_wiki_seed_{seed}.txt', 'w') as idname_w:
    #         idname_w.write('\n'.join(dico))
    #         idname_w.close()
    
    # all_dico = open(f'{write_dir}/image_classes_dict_wiki_seed_{seed}.txt').readlines()
    # all_dico_size = len(all_dico)

    # train_nums = math.ceil(all_dico_size * 0.7)
    # validation_nums = math.ceil(all_dico_size * 0.85)
    # test_nums = all_dico_size - validation_nums

    # with open(f'{write_dir}/train_wiki_dico_{seed}.txt', 'w+') as biw:
        
    #     for iword in all_dico[:train_nums]:
    #         biw.write(iword)
    #     biw.close()

    # with open(f'{write_dir}/eval_wiki_dico_{seed}.txt', 'w+') as biw:
    #     for iword in all_dico[train_nums:validation_nums]:
    #         biw.write(iword)
    #     biw.close()

    # with open(f'{write_dir}/test_wiki_dico_{seed}.txt', 'w+') as biw:
    #     for iword in all_dico[validation_nums:]:
    #         biw.write(iword)
    #     biw.close()

if __name__ == '__main__':
    ids_words_path = './data/image_ids_wiki_using.txt'
    write_dir = './data/dictionaries_id2w'

    if not os.path.exists(write_dir):
        os.mkdir(write_dir)

    # for _ in range(5):
    #     seed = random.randint(0, 100)
    #     dico_build(ids_words_path, seed)
    # seed = 'no'
    for _ in range(5):
        seed = random.randint(0, 1000)
        train_dico, eval_dico, test_dico = dico_build(ids_words_path, seed)

        dico_write(write_dir=write_dir, dicos=[train_dico, eval_dico, test_dico], seed=seed)
        # dico_write(write_dir=write_dir, dico=eval_dico, seed=seed)
        # dico_write(write_dir=write_dir, dico=test_dico, seed=seed)
