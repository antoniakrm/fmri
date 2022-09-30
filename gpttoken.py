from cProfile import label
from lib2to3.pgen2 import token
from transformers import GPT2TokenizerFast, GPT2Model
import numpy as np
import torch
import os
from tqdm import tqdm
from utils import *
from sentences_downloader import sentences_download

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_texts(wordslist, sentences_path):
    with open(sentences_path, 'r') as examples_read:
        contexts = examples_read.readlines()
        examples_read.close()
    with open(wordslist, 'r') as words_read:
        words = words_read.readlines()
        words_read.close()
        
    return words, contexts

def load_gpt2_model(string):
    model = GPT2Model.from_pretrained(string, output_hidden_states=True)
    tokenizer = GPT2TokenizerFast.from_pretrained(string, add_prefix_space=True, is_split_into_words=True)
    # tokenizer.add_special_tokens({'pad_token':'[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    return model.to(device), tokenizer

# Preparing the input for BERT
# Takes a string argument and performs
# pre-processing like adding special tokens, tokenization, 
# tokens to ids, and tokens to segment ids.
# All tokens are mapped to segment id = 0.
def gpt2_text_preparation(context, tokenizer):
    marked_text = f"<|endoftext|> {context} <|endoftext|>"
    tokenized_text = tokenizer.tokenize(marked_text)
    # batch_encoding = tokenizer(text, return_offsets_mapping=True)
    # indexed_tokens = batch_encoding['input_ids']
    # subwords_list = batch_encoding['offset_mapping']
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0]*len(indexed_tokens)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensor =  torch.tensor([segments_ids])

    return tokenized_text, tokens_tensor, segments_tensor

# Getting embeddings from an embedding model
def gpt2_embeddings(tokens_tensor, segments_tensors, model):
    with torch.no_grad():
        outputs = model(tokens_tensor)
        hidden_states = outputs[2][1:]
        # last_hidden_states = outputs.last_hidden_state

    last_hidden_state = hidden_states[-1]
    token_embeddings = torch.squeeze(last_hidden_state, dim=0)
    return [token_embed.tolist() for token_embed in token_embeddings]

# Getting the average word embeddings for
# the target word in all sentences
# Getting the average word embeddings for
# the target word in all sentences

def average_word_embedding(words, contexts, model, tokenizer):
    target_word_average_embeddings = []
    # words_size = subwords_size(words)
    for word in tqdm(words):
        word = word.strip('\n')
        word_no_dash = word.replace('_', ' ')
        # phrase_size = len(word.split('_'))
        # print(word, word_no_dash)
        tokenized_word = tokenizer.tokenize(word_no_dash)
        # print(tokenized_word)
        num_sub = len(tokenized_word) # Number of sub-words in a word
        subword_average_embedding = []
        for context in contexts:
            if word == context.split(':',1)[0].lower():
                context = context.split(': ',1)[1].lower().replace('"', ' ').replace('(', '( ').replace(')', ' )')
                tokenized_text, tokens_tensor, segments_tensors = gpt2_text_preparation(context.strip('\n'), tokenizer)
                list_token_embeddings = gpt2_embeddings(tokens_tensor.to(device), segments_tensors.to(device), model)
                # print(tokenized_text)
                # Get the first index of the word's token in the contexts
                # 'Ġevangelical'
                # if tokenized_word[0] in tokenized_text:
                try:
                    word_index = tokenized_text.index(tokenized_word[0])
                    if ''.join(tokenized_text[word_index: word_index + num_sub]) == ''.join(tokenized_word):
                    # word_embedding = list_token_embeddings[word_index: word_index + num_sub]
                        word_embedding = list_token_embeddings[word_index: word_index + num_sub]
                        subword_average_embedding.append(np.mean(word_embedding, axis=0))
                    # if f'Ġ{word_no_dash.split(" ")[0]}' in tokenized_text:
                    #     word_index = tokenized_text.index(tokenized_word[0])
                    #     word_embedding = list_token_embeddings[word_index: word_index + phrase_size]
                    # print(f"word index: {word_index}")
                    # Get the all subtokens of the word
                    # word_embedding = list_token_embeddings[word_index: word_index + num_sub]
                    # print(f"word_embedding_length: {len(word_embedding)}")

                    # average among the representations of these subtokens 
                except:
                    print(tokenized_word)
                
        # print(len(subword_average_embedding))

        # average among the representations of various contexts
        average_word_embedding = np.mean(subword_average_embedding, axis=0)
        target_word_average_embeddings.append(average_word_embedding)
        # print(average_word_embedding.shape)
    return np.array(target_word_average_embeddings)

def format_embeddings(X, words, embeddings_path):

  with open(f"{embeddings_path}.txt", 'w') as outfile:
    outfile.write(f'{str(X.shape[0])} {str(X.shape[1])}\n')
    for word, vec in zip(words, X):
      outfile.write(
          f"{word.strip().lower()} {' '.join([str(v) for v in vec.tolist()])}\n")
    outfile.close()

def format_train_eval(X, words, embeddings_path, training):
    if training != 'train':
        list_size = X.shape[0] - round(X.shape[0]*0.7)
        words = words[round(X.shape[0]*0.7):]
        X = X[round(X.shape[0]*0.7):]
    else:
        list_size = round(X.shape[0]*0.7)
        X = X[:list_size]
        words = words[:list_size]
    with open(f"{embeddings_path}_{training}.txt", 'w') as outfile:
        outfile.write(f'{str(list_size)} {str(X.shape[1])}\n')
        for word, vec in zip(words, X):
            outfile.write(
                f"{word.strip().lower()} {' '.join([str(v) for v in vec.tolist()])}\n"
            )
        outfile.close()

if __name__ == "__main__":

    parser = config_parser()
    args = parser.parse_args()
    logger = initialize_exp(args)
    if args.model_prefix:
        model = f'{args.model_prefix}/{args.model_name}'
    else:
        model = args.model_name
    # print(model)
    encodings_path = f"{args.output_dir}/{args.model_name}_encodings_new.npy"
    embeddings_path = f"{args.emb_dir}/{args.model_name}_new"

    sentences_download(args.download_path, args.wordlist_path, args.sentences_path)

    words, sentences = load_texts(wordslist=args.wordlist_path, sentences_path=args.sentences_path)

    if not os.path.exists(encodings_path):
        # print(words, sentences)
        model, tokenizer = load_gpt2_model(model)
        logger.info('==========Wording embedding==========')
        targets = average_word_embedding(words, sentences, model, tokenizer)
        logger.info(f'target shape: {targets.shape}')
        logger.info('==========Embedding complete==========')
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        if not os.path.exists(args.emb_dir):
            os.mkdir(args.emb_dir)
        np.save(encodings_path, targets)
    else:
        logger.info('Encodings file exists!')
        targets = np.load(encodings_path)
    logger.info('==========Format embeddings==========')
    format_embeddings(targets, words, embeddings_path)
    # format_train_eval(targets, words, embeddings_path, training='train')
    # format_train_eval(targets, words, embeddings_path, training='eval')
    logger.info('==========Format complete==========')
