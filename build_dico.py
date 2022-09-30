import os
import random
from itertools import product

# def dico_build(id_all_path, id_part_path, wordlist):
#     if not os.path.exists('./data/image_classes_dict_new.txt'):
#         lidname = open(id_all_path).readlines()
#         lpidname = open(id_part_path).readlines()
#         id_all = [i.strip().split(': ')[0] for i in lidname]
#         name_all = [i.strip().lower().split(': ')[1] for i in lidname]
#         id_p = [i.strip().split(': ')[0] for i in lpidname]
#         # name_p = [i.strip().lower().split(': ')[1].replace(' ', '_') for i in lpidname]

#         words_all = open(wordlist).read().strip().split('\n')

#         dico = []
#         for index, id in enumerate(id_p):
#             related_name = name_all[id_all.index(id)].split(', ')
#             related_name = filter(lambda x: x in words_all, related_name)
#             # for w in related_name:
#             #     if w == 'doctor':
#             #         print(related_name)
#             #     if w not in words_all:
#             #         related_name.remove(w)
#             relations = list(product(related_name, repeat=2))
#             # for name in related_name:
#             #     word_dico = f'{name_p[index]}    {name}'
#             #     dico.append(word_dico)
#             for src, tgt in relations:
#                 word_dico = f'{src}    {tgt}'
#                 dico.append(word_dico)

#         with open('./data/image_classes_dict_new.txt', 'w') as idname_w:
#             idname_w.write('\n'.join(dico))
#             idname_w.close()
    
#     all_dico = open('./data/image_classes_dict_new.txt').readlines()
#     all_dico_size = len(all_dico)

#     with open(f'./data/train_m_dico_new.txt', 'w+') as biw:
        
#             # biw.write(f'{str(round(images_size*0.7)-1)} {str(n)}\n')
#         for iword in all_dico[:round(all_dico_size*0.7)-1]:
#             # if iword in bert_words:
#             biw.write(iword)
#         print(round(all_dico_size*0.7)-1)
        
#             # biw.write(f'{str(images_size-round(images_size*0.7)+1)} {str(n)}\n')
#         biw.close()

#     with open(f'./data/eval_m_dico_new.txt', 'w+') as biw:
#         for iword in all_dico[round(all_dico_size*0.7)-1:]:
#             # if iword in bert_words:
#             biw.write(iword)
#         print(all_dico_size-round(all_dico_size*0.7)+1)
#             # biw.write(f'{str(images_size-round(images_size*0.7)+1)} {str(n)}\n')
#         biw.close()

def dico_build(ids_words_path, boundary):
    seed = 10
    # if not os.path.exists(f'./data/image_classes_dict_wiki_seed_{seed}.txt'):
    if not os.path.exists(f'./data/sub_words_{seed}.txt'):
        # lidname = open(id_all_path).readlines()
        random.seed(seed)
        lpidname = open(ids_words_path).readlines()
        # random.shuffle(lpidname)
        id_all = [i.strip().split(': ')[0] for i in lpidname]
        name_all = [i.strip().lower().split(': ')[1] for i in lpidname]
        
        dico = []
        related_dict = {}
        for id_v in id_all:
            related_names = []
            for idx, id_val in enumerate(id_all):
                if id_v == id_val:
                    name = name_all[idx]
            # if id == id[idx+1]:
                    related_names.append(name)
            related_dict[id_v] = related_names
        
        keys = list(related_dict.keys())
        random.shuffle(keys)
        shuffled_dict = dict()
        for key in keys:
            shuffled_dict.update({key: related_dict[key]})

        for value in shuffled_dict.values():
            relations = list(product(value, repeat=2))
            for src, tgt in relations:
                word_dico = f'{src}    {tgt}'
                dico.append(word_dico)
        
            # for w in related_name:
            #     if w == 'doctor':
            #         print(related_name)
            #     if w not in words_all:
            #         related_name.remove(w)
            # relations = list(product(related_name, repeat=2))
            # # for name in related_name:
            # #     word_dico = f'{name_p[index]}    {name}'
            # #     dico.append(word_dico)
            # for src, tgt in relations:
            #     word_dico = f'{src}    {tgt}'
            #     dico.append(word_dico)

        # with open(f'./data/image_classes_dict_wiki_seed_{seed}.txt', 'w') as idname_w:
        with open('./data/sub_words.txt','w') as idname_w:
            idname_w.write('\n'.join(dico))
            idname_w.close()
    
    # all_dico = open(f'./data/image_classes_dict_wiki_seed_{seed}.txt').readlines()
    all_dico = open('./data/sub_words.txt').readlines()
    # all_dico_size = len(all_dico)
    src_words = [i.split()[0] for i in all_dico]

    count = 0
    for idx, word in enumerate(src_words):
        if idx < len(src_words)-1 and word != src_words[idx+1]:
            count += 1
        if count == boundary:
            # print(count)
            border = idx
            break

    with open(f'./data/words_dict_sub/train_wiki_dico_{seed}_{boundary}.txt', 'w+') as biw:
        
            # biw.write(f'{str(round(images_size*0.7)-1)} {str(n)}\n')
        for iword in all_dico[:border+1]:
            # if iword in bert_words:
            biw.write(iword)
        # print(round(all_dico_size*0.7)-1)
        
            # biw.write(f'{str(images_size-round(images_size*0.7)+1)} {str(n)}\n')
        biw.close()

    with open(f'./data/words_dict_sub/eval_wiki_dico_{seed}_{boundary}.txt', 'w+') as biw:
        # for iword in all_dico[round(all_dico_size*0.7)-1:]:
        for iword in all_dico[border+1:]:
            # if iword in bert_words:
            biw.write(iword)
        # print(all_dico_size-round(all_dico_size*0.7)+1)
            # biw.write(f'{str(images_size-round(images_size*0.7)+1)} {str(n)}\n')
        biw.close()

# def load_words(path, training):
#     image_classes_words = [i.strip('\n').split(': ', 1)[1].lower().replace(' ','_') for i in open(path).readlines()]
#     images_size = len(image_classes_words)
#     with open(f'./data/{training}_dico.txt', 'w+') as biw:
#         if training == 'train_m':
#             # biw.write(f'{str(round(images_size*0.7)-1)} {str(n)}\n')
#             for iword in image_classes_words[:round(images_size*0.7)-1]:
#                 # if iword in bert_words:
#                 biw.write(f'{iword}    {iword}\n')
#             print(round(images_size*0.7)-1)
#         else:
#             # biw.write(f'{str(images_size-round(images_size*0.7)+1)} {str(n)}\n')
#             for iword in image_classes_words[round(images_size*0.7)-1:]:
#                 # if iword in bert_words:
#                 biw.write(f'{iword}    {iword}\n')
#             print(images_size-round(images_size*0.7)+1)
#         biw.close()

if __name__ == '__main__':
    # ids_words_path = './data/image_ids_wiki_using.txt'
    ids_words_path = './data/sub_image_ids_wiki_using.txt'
    # wordlist = './data/wordlist_ordered.txt'
    for i in [10, 50, 100, 500, 910]:
        boundary = i
        dico_build(ids_words_path, boundary)
    # load_words('data/outputs/real_images_resnet50_image_classes.txt', 'train')
    # load_words('./data/image_id_over100.txt', 'eval_m')
    # load_words('./data/image_id_over100.txt', 'train_m')
    # load_words('data/outputs/real_images_resnet50_image_classes.txt', 'eval')