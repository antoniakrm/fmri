import json
import os
import numpy as np

def get_frequency_rank():
    word2freq = json.load(open('/image/nlp-datasets/yova/freq_wordlist_nltk'))
    words_list = open('./data/wordlist_ordered.txt').read().strip().lower().split('\n')
    print('length of words_list: ', len(words_list))
    all_words = list(word2freq.keys())
    all_freqs = [word2freq[w] for w in all_words]
    freq_sort = np.argsort(all_freqs)[::-1]
    all_words_sorted = [all_words[i] for i in freq_sort]
    word2rank = {word: i for i, word in enumerate(all_words_sorted)}
    # test = word2freq['basketball']

    eval_word_rank = {}
    for word in words_list:
        if word not in word2rank:
            eval_word_rank[word] = len(word2rank) + 1
        else:
            eval_word_rank[word] = word2rank[word]
    return eval_word_rank

def build_frequency_dict(eval_word_rank, save_dir):
    # model = 'seg'
    # dis_path =  os.path.expanduser(f"~/Dir/projects/IPLVE/data/sorted_dispersion_{model}.txt")
    # dis_sorted = open(dis_path).readlines()
    # dis_sorted_info = [i.split(': ', 1)[0] for i in dis_sorted]

    dict_path = './data/dictionaries_id2w'
    dict_list = os.listdir(dict_path)
    seeds = [203, 255, 633, 813, 881]
    for seed in seeds:
        # bin500 = []
        bin5000 = []
        bin50000 = []
        bin_other = []
        bin_phrase = []
        bins = [bin5000, bin50000, bin_other, bin_phrase]
        for dico in dict_list:
            if "eval" in dico and str(seed) in dico:
                dictionaries = open(os.path.join(dict_path, dico)).readlines()

                # eval_ids_origin = [i.split(' ', 1)[0] for i in dictionaries]
                eval_words = [i.strip().split(' ', 1)[1] for i in dictionaries]

                for idx, word in enumerate(eval_words):
                    if len(word.split('_')) > 1:
                        bins[3].append(dictionaries[idx])
                    # elif eval_word_rank[word] < 500:
                    #     bins[0].append(dictionaries[idx])
                    elif 5000 > eval_word_rank[word]:
                        bins[0].append(dictionaries[idx])
                    elif 5000 <= eval_word_rank[word] < 50000:
                        bins[1].append(dictionaries[idx])
                    elif 50000 <= eval_word_rank[word]:
                        bins[2].append(dictionaries[idx])                  


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

                print(len(bin5000), len(bin50000), len(bin_other), len(bin_phrase))
                for block_idx, block_name in enumerate(['5k', '50k', 'others', 'phrase']):
                    with open(f'{save_dir}/eval_{seed}_{block_name}.txt', 'w') as wd:
                        for pair in bins[block_idx]:
                            wd.write(pair)
                        wd.close()
                        # wd.write('\n'.join(sorted_res))

# 151 581 778 795
# 157 635 769 766
# 160 658 737 833
# 170 628 739 787
# 143 594 782 829

if __name__ == "__main__":
    res = get_frequency_rank()
    save_dir = './data/dictionaries_frequency2'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    build_frequency_dict(res, save_dir)
                


