# import string
# from transformers import BertTokenizerFast, BertModel
# import pandas as pd
import numpy as np
import os
# import nltk
# import torch
import requests
import json
# from language_detector import detect_language
# import unicodedata
# from bs4 import BeautifulSoup
# import configargparse
import fasttext as ft
import threading

ft.FastText.eprint = lambda x:None

class Crawl_sentences(threading.Thread):
    def __init__(self, threadID, name, wordlist, index, download_path):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.index = index
        self.name = name
        self.download_path = download_path
        self.wordlist = wordlist
    def run(self):
        print(f"Begin: {self.name}")
        create_sentences_file(self.wordlist, self.index, self.threadID, self.download_path)
        print(f"End: {self.name}")

def is_english(text, model):
    return model.predict(text)[0][0] == '__label__en'

def check_symbols(s):
    arr = []
    SYMBOLS = {'}': '{', ']': '[', ')': '(', '>': '<'}
    SYMBOLS_L, SYMBOLS_R = SYMBOLS.values(), SYMBOLS.keys()
    for c in s:
        if c in SYMBOLS_L:
            # push symbol left to list
            arr.append(c)
        elif c in SYMBOLS_R:
            # pop out symbol,
            if arr and arr[-1] == SYMBOLS[c]:
                arr.pop()
            else:
                return False
    return True

def crawl_sentences(word, model):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 SE 2.X MetaSr 1.0'}

    sentence_list = []
    # ft_model = ft.load_model("./pretrained/lid.176.bin")

    try:
        r = requests.get(f'https://api.rhymezone.com/words?max=501&nonorm=1&k=rz_wke&rel_wke={word}', headers=headers).text

        if len(r) > 2:
            try:
                start = r.index('[{')
                end = r.index('}]') + 2
                data = json.loads(r[start:end])
            except Exception:
                print(f"loaded error: {word}")
            for item in data:
                item_info = item['word'].replace('<b>', '').replace('</b>', '').split(':', 3)
                # print(item_info)
                # Only select the wiki source
                item_sentence = item_info[-1]
                # if (item_info[0] == 'd' or item_info[0] == 'b' or item_info[0] == 'q')
                if (item_info[0] == 'd') \
                        and len(item_sentence.split(' ')) < 16 and '...' not in item_sentence\
                        and check_symbols(item_sentence) and item_sentence.count('"')%2 == 0 \
                        and item_sentence.isascii() and is_english(item_sentence, model): 
                    source = f'{word}: {item_sentence}'
                    sentence_list.append(source)

    except Exception:
        print(f"request error:{word}")
    return sentence_list

def create_sentences_file(wordslist, index, part, download_path):
    with open(wordslist, 'r') as words_read:
        words = words_read.readlines()[index:index+2400]
        examples = []
        ft_model = ft.load_model("./pretrained/lid.176.bin")
        for word_n in words:
            word = word_n.strip('\n').replace(' ','_')
            # print(word)

            crawled_sentences = crawl_sentences(word=word, model=ft_model)
            if len(crawled_sentences) >= 5:
                examples.append(crawled_sentences[:15])
            # else:
            #     words.remove(word_n)
        words_read.close()
        if examples:
            if not os.path.exists(download_path):
                os.mkdir(download_path)
            with open(f'{download_path}/sentences_{part}.txt', 'w') as sentences_write:
                for sentences in examples:
                    for sentence in sentences:
                        sentences_write.write(sentence + '\n')
                sentences_write.close()

# def config_parser():
#     # import configargparse
#     parser = configargparse.ArgumentParser()
#     parser.add_argument('--config', is_config_file=True,
#                         help='config file path')
#     parser.add_argument("--wordlist_path", type=str, default="./data/wordlist.txt",
#                         help='the whole wordlist file path')
#     parser.add_argument("--sentences_path", type=str, default="./data/sentences.txt",
#                         help='sentences file path')
#     # parser.add_argument("--encodings file path", type=string, default="",
#     #                     help='embeddings folder path')
#     # parser.add_argument("--encodings file path", type)
#     return parser


def sum_files(root_path, sentences_path):
    files = os.listdir(root_path)
    words_repeated = []
    sentences = []
    for file in files:
        with open(f'{root_path}/{file}', 'r') as file_reader:
            file_sentences = file_reader.readlines()
            sentences += file_sentences
            for sentence in file_sentences:
                word = sentence.split(":", 1)[0]
                # print(word)
                words_repeated.append(word)

    words = list(dict.fromkeys(words_repeated))
    sentences_no_repeat = list(dict.fromkeys(sentences))

    with open('./data/wordlist_satisfied.txt','w+') as word_w:
        for word in words:
            word_w.write(f'{word}\n')
        word_w.close()

    with open(sentences_path,'w') as sent_w:
        for sent in sentences_no_repeat:
            sent_w.write(sent.strip('\n')+'\n')
        sent_w.close()


def build_image_classes_maps():
    words = open('./data/wordlist_satisfied.txt').read().strip().split('\n')
    image_classes_ids = os.listdir('./data/imagenet_21k_small')

    image_maps = open('./data/imagenet21k_ids_names.txt').read().strip().split('\n')
    image_dict = {}
    for image_map in image_maps:
        image_dict[image_map.split(': ')[0]] = image_map.split(': ')[1].split(', ')

    count = 0
    # image_with_sentences = []
    image_ids_using = []
    image_words_using = []
    for id in image_classes_ids:
        curr = list(filter(lambda x: x in words, image_dict[id]))
        if curr:
            count += 1
            for w in curr:
                image_ids_using.append(id)
                image_words_using.append(w)
            # image_with_sentences.extend(curr)
    print(count)

    with open('./data/image_ids_wiki_using.txt', 'w+') as ids_w:
        for x,y in zip(image_ids_using, image_words_using):
            ids_w.write(f'{x}: {y}\n')
        ids_w.close()


def sentences_download(download_path, wordlist, sentences_path):
    # parser = config_parser()
    # args = parser.parse_args()
    if not os.path.exists(sentences_path):
        print('Begin downloading sentences.')
        threads = []
        for i in range(60):
            thread_index = Crawl_sentences(i, f"Thread-{i}", wordlist=wordlist, index=i*2400, download_path=download_path)
            threads.append(thread_index)

        for thre in threads:
            thre.start()

        for thre in threads:
            thre.join()
        sum_files(download_path, sentences_path)
        build_image_classes_maps()
    print('Download sentences successfully.')