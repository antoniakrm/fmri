from transformers import BertTokenizerFast, BertModel
import numpy as np
import torch
import os
from tqdm import tqdm
from utils import *

# An orrange tabby cat is sleeping on the floor.

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_bert_model(string):
    model = BertModel.from_pretrained(string, \
        cache_dir=os.path.expanduser(f"~/.cache/huggingface/transformers/models/{string}"), output_hidden_states=True)
    tokenizer = BertTokenizerFast.from_pretrained(string)
    return model.to(device), tokenizer

# Preparing the input for BERT
# Takes a string argument and performs
# pre-processing like adding special tokens, tokenization, 
# tokens to ids, and tokens to segment ids.
# All tokens are mapped to segment id = 0.
def bert_text_preparation(context, tokenizer):
    marked_text = f"[CLS] {context} [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    # batch_encoding = tokenizer(text, return_offsets_mapping=True)
    # indexed_tokens = batch_encoding['input_ids']
    # subwords_list = batch_encoding['offset_mapping']
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0]*len(indexed_tokens)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensor =  torch.tensor([segments_ids])

    return tokenized_text, tokens_tensor.to(device), segments_tensor.to(device)

# Getting embeddings from an embedding model
def bert_embeddings(tokens_tensor, segments_tensors, model):
    with torch.no_grad():
        outputs = model(input_ids=tokens_tensor, token_type_ids=segments_tensors)
        # outputs = model(tokens_tensor)
        # hidden_states = outputs[2][1:]

    # last_hidden_state = hidden_states[-1]
    last_hidden_state = outputs.last_hidden_state
    token_embeddings = torch.squeeze(last_hidden_state, dim=0)
    return [token_embed.tolist() for token_embed in token_embeddings]


model, tokenizer = load_bert_model('google/bert_uncased_L-8_H-512_A-8')
sentences = ['A sport car is driving down a road.', 
'A sport car is being washed at a carwash.',
'A red sport car.',
'A blue sport car.',
'A sport car is parked in the middle of a green field.',
'A sport car is falling off a cliff.']

phrase = 'sport car'
tokenized_word = tokenizer.tokenize(phrase)
num_sub = len(tokenized_word)
subword_average_embedding = []
for sentence in sentences:
# sentence = 'A sport car is parked in the middle of a green field.'
    tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(sentence, tokenizer)
    list_token_embeddings = bert_embeddings(tokens_tensor, segments_tensors, model)
    word_indices = [i for i,x in enumerate(tokenized_text) if x == tokenized_word[0]]
    for index_word in word_indices:
        word_embedding = list_token_embeddings[index_word: index_word + num_sub]
        if tokenized_word == tokenized_text[index_word: index_word + num_sub]:
            # if tokenized_word_size[index_in_phrase] > 1:
            #     words_embeddings.append(np.mean([div_norm(np.array(i)) for i in word_embedding], axis=0))
            # else:
            # words_embeddings.append(np.mean(word_embedding, axis=0))
            subword_average_embedding.append(word_embedding)
            break

print(len(subword_average_embedding))
with open(os.path.expanduser('~/Dir/projects/summer_intern/data_words_images/sentences_embeddings_512.txt'), 'w+') as sw:
    sw.write(f'6 512\n')
    for i in range(6):
        vec = subword_average_embedding[i][0]
        sw.write(f"{f'sport_car_{i}'.lower().replace(' ', '_')} {' '.join([str(v) for v in vec])}\n")

