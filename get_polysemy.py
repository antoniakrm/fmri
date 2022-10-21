import json
import os

def get_polyseme():
    cache = json.load(open('/image/nlp-datasets/yova/cached_requests_omw'))
    words_list = open('./data/wordlist_ordered.txt').read().strip().lower().split('\n')
    print('length of words_list: ', len(words_list))
    eng_table = cache['en']
    # test = eng_table['drawer'][0]
    test = eng_table['towel'][0]
    # for en in eng_table:
    #     size = len(en[0].split('_'))
    #     if size > 1:
    #         print(en[0])
    # test2 = eng_table['rivet'][0]
    num_meaning = {}
    for word in words_list:
        if len(word.split('_')) > 1:
            num_meaning[word] = 'phrase' 
        elif word not in eng_table:
            num_meaning[word] = -1
        else:
            num_meaning[word] = eng_table[word][0]
    return num_meaning

def build_polyseme_dict(num_meaning, save_dir):
    # model = 'seg'
    # dis_path =  os.path.expanduser(f"~/Dir/projects/IPLVE/data/sorted_dispersion_{model}.txt")
    # dis_sorted = open(dis_path).readlines()
    # dis_sorted_info = [i.split(': ', 1)[0] for i in dis_sorted]

    dict_path = './data/dictionaries_id2w'
    dict_list = os.listdir(dict_path)
    seeds = [203, 255, 633, 813, 881]
    for seed in seeds:
        bin1 = []
        bin2_3 = []
        bin_over_3 = []
        bin_phrase = []
        bin_unk = []
        bins = [bin1, bin2_3, bin_over_3, bin_phrase, bin_unk]
        for dico in dict_list:
            if "eval" in dico and str(seed) in dico:
                dictionaries = open(os.path.join(dict_path, dico)).readlines()

                # eval_ids_origin = [i.split(' ', 1)[0] for i in dictionaries]
                eval_words = [i.strip().split(' ', 1)[1] for i in dictionaries]

                for idx, word in enumerate(eval_words):
                    if num_meaning[word] == "phrase":
                        bins[3].append(dictionaries[idx])
                    elif num_meaning[word] == -1:
                        bins[4].append(dictionaries[idx])
                    elif num_meaning[word] > 3:
                        bins[2].append(dictionaries[idx])
                    elif num_meaning[word] == 1:
                        bins[0].append(dictionaries[idx])
                    else:
                        bins[1].append(dictionaries[idx])


                # sorted_index = [dis_sorted_info.index(i) for i in eval_words]
                # eval_zipped = list(zip(eval_ids_origin, eval_words, sorted_index))
                # sorted_res = sorted(eval_zipped, key=operator.itemgetter(2))
                # # eval_zipped.sort(key=dis_sorted_ids.index)
                # block_size = math.ceil(len(sorted_res) / 3)
                # id_final, word_final, _ = zip(*sorted_res)
                # # id_final = list(id_final)
                # # word_final = list(word_final)
                # final_list = list(zip(id_final, word_final))
                # print(final_list[0:2])

                print(len(bin1), len(bin2_3), len(bin_over_3), len(bin_phrase), len(bin_unk))
                for block_idx, block_name in enumerate(['single', '2to3', 'over3', 'phrase','unk']):
                    with open(f'{save_dir}/eval_{seed}_{block_name}.txt', 'w') as wd:
                        for pair in bins[block_idx]:
                            wd.write(pair)
                        wd.close()

if __name__ == "__main__":
    res = get_polyseme()
    save_dir = './data/dictionaries_polysemy'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    build_polyseme_dict(res, save_dir)
                


