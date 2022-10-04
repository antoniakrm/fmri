from transformers import BertTokenizerFast, BertModel
import numpy as np
import torch
import os
from tqdm import tqdm
from modified_utils import *
from sentences_downloader import sentences_download

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

def load_texts(download, wordslist, sentences_path):

    # sentences_download(args.download_path, args.wordlist_path, args.sentences_path)
    sentences_download(download, wordslist, sentences_path)
    with open(sentences_path, 'r') as examples_read:
        contexts = examples_read.readlines()
        examples_read.close()
    with open(wordslist, 'r') as words_read:
        words = words_read.readlines()
        words_read.close()
        
    return words, contexts

def load_bert_model(string):
    model = BertModel.from_pretrained(string, \
        cache_dir=os.path.expanduser(f"~/.cache/huggingface/transformers/models/{args.model_name}"), output_hidden_states=True)
    tokenizer = BertTokenizerFast.from_pretrained(string)
    return model.to(device), tokenizer

# def l2_norm(x):
#    return np.sqrt(np.sum(x**2))

# def div_norm(x):
#    norm_value = l2_norm(x)
#    if norm_value > 0:
#        return x * ( 1.0 / norm_value)
#    else:
#        return x

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

# Getting the average word embeddings for
# the target word in all sentences
# Getting the average word embeddings for
# the target word in all sentences

def average_word_embedding(contexts, model, tokenizer):
    iters = tqdm(contexts, mininterval=300.0, maxinterval=600.0)
    words_order = [i.split(':',1)[0].lower() for i in contexts] # size 106564
    
    word_current = words_order[0]
    subword_average_embedding = []
    target_word_average_embeddings = []
    words_in_sentences = [word_current]

    for ids, context in enumerate(iters):
        # if ids % int(1517786/100) == 0:
        # iters.set_description(f'Target shape: {np.array(target_word_average_embeddings).shape}')
        if word_current == words_order[ids]:
            phrase = word_current.replace('_', ' ')
            # phrase_size = len(phrase)

            tokenized_word = tokenizer.tokenize(phrase)
            num_sub = len(tokenized_word)
            context = context.split(': ',1)[1].strip('\n')
            tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(context.strip('\n'), tokenizer)
            list_token_embeddings = bert_embeddings(tokens_tensor, segments_tensors, model)
            # Get the first index of the word's token in the contexts
            # words_embeddings = []

            # for index_in_phrase in range(phrase_size):
            word_indices = [i for i,x in enumerate(tokenized_text) if x == tokenized_word[0]]
            for index_word in word_indices:
                word_embedding = list_token_embeddings[index_word: index_word + num_sub]
                if tokenized_word == tokenized_text[index_word: index_word + num_sub]:
                    # if tokenized_word_size[index_in_phrase] > 1:
                    #     words_embeddings.append(np.mean([div_norm(np.array(i)) for i in word_embedding], axis=0))
                    # else:
                    # words_embeddings.append(np.mean(word_embedding, axis=0))
                    subword_average_embedding.append(np.mean(word_embedding, axis=0))
                    break
            # Get the all subtokens of the word

            # average among the representations of these subtokens 
            # if phrase_size > 1:
            #     subword_average_embedding.append(np.mean([div_norm(np.array(i)) for i in words_embeddings], axis=0))
            # else:
            # subword_average_embedding.append(np.mean(words_embeddings, axis=0))

        if ids == len(contexts)-1:
            average_word_embedding = np.mean(subword_average_embedding, axis=0)
            target_word_average_embeddings.append(average_word_embedding)
            subword_average_embedding = []
            return words_in_sentences, np.array(target_word_average_embeddings)

        if words_order[ids+1] != words_order[ids]:
            average_word_embedding = np.mean(subword_average_embedding, axis=0)
            target_word_average_embeddings.append(average_word_embedding)
            subword_average_embedding = []
            word_current = words_order[ids+1]
            words_in_sentences.append(word_current)

if __name__ == "__main__":

    parser = config_parser()
    args = parser.parse_args()
    logger = initialize_exp(args)
    if args.model_prefix:
        model = f'{args.model_prefix}/{args.model_name}'
    else:
        model = args.model_name
    # print(model)
    encodings_path = f"{args.output_dir}/{args.model_name}_encodings.npy"
    embeddings_path = f"{args.emb_dir}/{args.model_name}"

    words, sentences = load_texts(wordslist=args.wordlist_path, sentences_path=args.sentences_path, download=args.download_path)

    sentences_download(args.download_path, args.wordlist_path, args.sentences_path)

    if not os.path.exists(encodings_path):
        # print(words, sentences)
        model, tokenizer = load_bert_model(model)
        logger.info('==========Wording embedding==========')
        words_in_sentences, targets = average_word_embedding(sentences, model, tokenizer)
        logger.info(f'target shape: {targets.shape}')
        with open(args.ordered_words_path,'w') as word_w:
            for word in words_in_sentences:
                word_w.write(f'{word}\n')
            word_w.close()
        logger.info('==========Embedding complete==========')
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        if not os.path.exists(args.emb_dir):
            os.makedirs(args.emb_dir)
        np.save(encodings_path, targets)
    else:
        logger.info('Encodings file exists!')
        targets = np.load(encodings_path)
        words_in_sentences = open(args.ordered_words_path).read().strip().lower().split('\n')
    logger.info('==========Format embeddings==========')
    # format_embeddings(targets, words_in_sentences, embeddings_path)
    if args.n_components < targets.shape[1]:
        output_reduced_dir = os.path.expanduser(f"~/Dir/projects/IPLVE/data/outputs/LM_out_reduced")
        embeddings_reduced_dir = os.path.expanduser(f"~/Dir/projects/IPLVE/data/embeddings/LM_emb_reduced")

        if not os.path.exists(output_reduced_dir):
            os.makedirs(output_reduced_dir)
        if not os.path.exists(embeddings_reduced_dir):
            os.makedirs(embeddings_reduced_dir)

        reduced_encodings_path = f"{output_reduced_dir}/{args.model_name}_encodings_reduced_{args.n_components}.npy"
        embeddings_path = f"{embeddings_reduced_dir}/{args.model_name}_reduced_{args.n_components}"
        
        logger.info('==========Reducing dimensionality==========')
        final_target = reduce_encoding_size(targets, reduced_encodings_path, args.n_components)
        logger.info('==========Reducing dimensionality is completed!==========')
    else:
        final_target = targets
    
    format_embeddings(final_target, words_in_sentences, embeddings_path)

    logger.info('==========Format complete==========')
