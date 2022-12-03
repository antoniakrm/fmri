import os
import subprocess
from datetime import datetime as dt

import numpy as np
import torch
from sklearn.decomposition import PCA

from MUSE.src.logger import create_logger

from .sentences_downloader import sentences_download


def initialize_exp(params):
    MAIN_DUMP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'dumped')
    # create the main dump path if it does not exist

    exp_folder = MAIN_DUMP_PATH if params.expdir.expname == '' else params.expdir.expname
    if not os.path.exists(exp_folder):
        subprocess.Popen("mkdir %s" % f'{exp_folder}', shell=True).wait()
    # params.log_path = exp_folder
    logger = create_logger(os.path.join(exp_folder, f'{dt.now().strftime("%Y%m%d%H%M%S")}.log'), vb=1)
    logger.info('============ Initialized logger ============')
    strings = ''
    for _, d in params.items():
        strings += '\n'.join(f'{k}: {str(v)}' for k, v in sorted(d.items())) + "\n"
    logger.info(strings)
    logger.info(f'The experiment will be stored in {exp_folder}')
    return logger

def format_embeddings(X, words, embeddings_path):

  with open(f"{embeddings_path}.txt", 'w') as outfile:
    outfile.write(f'{str(X.shape[0])} {str(X.shape[1])}\n')
    for word, vec in zip(words, X):
      outfile.write(
          f"{word.strip().lower()} {' '.join([str(v) for v in vec.tolist()])}\n")
    outfile.close()

def reduce_encoding_size(X, reduced_encodings_path, n_components=128):

    if os.path.exists(reduced_encodings_path):
        return np.load(reduced_encodings_path)
    print('Original Shape: ', X.shape)
    pca = PCA(n_components=n_components)
    tr_data = pca.fit_transform(X)
    print('Reduced Shape: ', tr_data.shape)

    np.save(reduced_encodings_path, tr_data)

    return tr_data

# Download the sentences related to the words in the wordlist
def load_texts(args):
    wordslist = args.data.wordlist_path
    sentences_path = args.data.sentences_path
    # sentences_download(args.download_path, args.wordlist_path, args.sentences_path)
    sentences_download(args)
    with open(sentences_path, 'r') as examples_read:
        contexts = examples_read.readlines()
        examples_read.close()
    with open(wordslist, 'r') as words_read:
        words = words_read.readlines()
        words_read.close()
        
    return words, contexts


# Getting embeddings from an embedding model
def get_embeddings(inputs, model):
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    token_embeddings = torch.squeeze(last_hidden_state, dim=0)
    return [token_embed.tolist() for token_embed in token_embeddings]