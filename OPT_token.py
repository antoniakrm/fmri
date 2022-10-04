from transformers import GPT2Tokenizer, OPTModel
import numpy as np
import torch
import os
from tqdm import tqdm
from encode_utils import *
from sentences_downloader import sentences_download

if not torch.cuda.is_available():
    device_ids = ["cpu"]
else:
    device_ids = [i for i in range(torch.cuda.device_count())]

def load_texts(download, wordslist, sentences_path):

    sentences_download(args.download_path, args.wordlist_path, args.sentences_path)
    sentences_download(download, wordslist, sentences_path)
    with open(sentences_path, 'r') as examples_read:
        contexts = examples_read.readlines()
        examples_read.close()
    with open(wordslist, 'r') as words_read:
        words = words_read.readlines()
        words_read.close()
        
    return words, contexts

def load_opt_model(string):

    device_map = None if len(device_ids)==1 else "auto"

    if string == "facebook/opt-350m":
        model = OPTModel.from_pretrained("facebook/opt-350m",\
            cache_dir=os.path.expanduser(f"~/.cache/huggingface/transformers/models/{args.model_name}"),\
            ignore_mismatched_sizes=True, word_embed_proj_dim=1024)
    else:
        model = OPTModel.from_pretrained(string, \
            cache_dir=os.path.expanduser(f"~/.cache/huggingface/transformers/models/{args.model_name}"), \
                device_map=device_map)
        
    tokenizer = GPT2Tokenizer.from_pretrained(string)
    # return model.to(device), tokenizer
    if not device_map:
        model = model.to(device_ids[0])
    return model, tokenizer


# Preparing the input for BERT
# Takes a string argument and performs
# pre-processing like adding special tokens, tokenization, 
# tokens to ids, and tokens to segment ids.
# All tokens are mapped to segment id = 0.
def opt_text_preparation(context, tokenizer):

    tokenized_text = tokenizer.tokenize(f'</s>{context}')
    inputs = tokenizer(context, return_tensors="pt")

    # tt = tokenizer(context)['input_ids']
    # test_convert = tokenizer.convert_tokens_to_ids(tokenized_text)
    # print(tt == test_convert)

    return tokenized_text, inputs.to(device_ids[0])
    # return tokenized_text, inputs

# Getting embeddings from an embedding model

def opt_embeddings(inputs, model):
    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state
    token_embeddings = torch.squeeze(last_hidden_state, dim=0)
    return [token_embed.tolist() for token_embed in token_embeddings]

# Getting the average word embeddings for
# the target word in all sentences
# Getting the average word embeddings for
# the target word in all sentences
# @jit
def average_word_embedding(contexts, model, tokenizer):
    iters = tqdm(contexts, mininterval=1800.0, maxinterval=3600.0)
    # iters = tqdm(contexts,mininterval=2, maxinterval=4)
    words_order = [i.split(':',1)[0].lower() for i in contexts]
    
    word_current = words_order[0]
    subword_average_embedding = []
    target_word_average_embeddings = []
    words_in_sentences = [word_current]

    for ids, context in enumerate(iters):
        if word_current == words_order[ids]:
            phrase_lower = word_current[:-2].replace('_', ' ')
            context = context.split(': ',1)[1].strip('\n')

            # Find the spercific type of the phrase in the sentence
            # Find the correct index of the tokens in the sentence
            phrase_pos = context.lower().find(phrase_lower)
            tokenized_text, inputs = opt_text_preparation(context, tokenizer)
            list_token_embeddings = opt_embeddings(inputs, model)

            # The word is the first word
            if phrase_pos == 0:
                phrase_type = 0
                phrase = f'{context[phrase_pos : phrase_pos + len(phrase_lower)]}'
                tokenized_word = tokenizer.tokenize(phrase)
                num_sub = len(tokenized_word)
            # The word is in "" or ()
            elif context[phrase_pos-1] == '(' or context[phrase_pos-1] == '"':
                phrase_type = 1
                phrase =  f'({context[phrase_pos : phrase_pos + len(phrase_lower)]}'
                tokenized_word = tokenizer.tokenize(phrase)
                num_sub = len(tokenized_word)-1
            # The word is not the first word
            else:
                phrase_type = 0
                phrase = f' {context[phrase_pos : phrase_pos + len(phrase_lower)]}'
                tokenized_word = tokenizer.tokenize(phrase)
                num_sub = len(tokenized_word)

            # Get the first index of the word's token in the contexts
            word_indices = [i for i,x in enumerate(tokenized_text) if x == tokenized_word[phrase_type]]
            for index_word in word_indices:
                word_embedding = list_token_embeddings[index_word: index_word + num_sub]
                text_contents = tokenized_text[index_word: index_word + num_sub]
                if tokenized_word[phrase_type:] == text_contents:
                    subword_average_embedding.append(np.mean(word_embedding, axis=0))
                    break
            # if phrase_type == 0:
            #     word_indices = [i for i,x in enumerate(tokenized_text) if x == tokenized_word[1]]
            #     for index_word in word_indices:
            #         word_embedding = list_token_embeddings[index_word: index_word + num_sub]
            #         if tokenized_word[1:-1] == tokenized_text[index_word: index_word + num_sub]:
            #             subword_average_embedding.append(np.mean(word_embedding, axis=0))
            #             break
            # else:
            #     word_indices = [i for i,x in enumerate(tokenized_text) if x == tokenized_word[0]]
            #     for index_word in word_indices:
            #         word_embedding = list_token_embeddings[index_word: index_word + num_sub]
            #         if tokenized_word == tokenized_text[index_word: index_word + num_sub]:
            #             subword_average_embedding.append(np.mean(word_embedding, axis=0))
            #             break

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
        # if ids%1000 == 0:
        #     logger.info(f'{ids} finished')


if __name__ == "__main__":

    # print(f'devices: {device_ids}')
    # os.system("nvidia-smi")
    
    parser = config_parser()
    args = parser.parse_args()
    # os.makedirs(cache_path) if not os.path.exists(cache_path) else None
    # os.environ['TRANSFORMERS_CACHE'] = cache_path
    logger = initialize_exp(args)
    if args.model_prefix:
        model = f'{args.model_prefix}/{args.model_name}'
    else:
        model = args.model_name
    # print(model)
    encodings_path = f"{args.output_dir}/{args.model_name.replace('.','-')}_encodings.npy"
    embeddings_path = f"{args.emb_dir}/{args.model_name.replace('.','-')}"

    words, sentences = load_texts(wordslist=args.wordlist_path, sentences_path=args.sentences_path, download=args.download_path)

    # sentences_download(args.download_path, args.wordlist_path, args.sentences_path)

    if not os.path.exists(encodings_path):
        # print(words, sentences)
        model, tokenizer = load_opt_model(model)
        logger.info('==========Wording embedding==========')
        # os.system("nvidia-smi")
        words_in_sentences, targets = average_word_embedding(sentences, model, tokenizer)
        logger.info(f'target shape: {targets.shape}')
        with open(args.ordered_words_path,'w') as word_w:
            for word in words_in_sentences:
                word_w.write(f'{word}\n')
            word_w.close()
        logger.info('==========Embedding complete==========')
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        if not os.path.exists(args.emb_dir):
            os.mkdir(args.emb_dir)
        np.save(encodings_path, targets)
    else:
        logger.info('Encodings file exists!')
        targets = np.load(encodings_path)
        words_in_sentences = open(args.ordered_words_path).read().strip().lower().split('\n')
    logger.info('==========Format embeddings==========')

    if args.n_components < targets.shape[1]:
        output_reduced_dir = os.path.expanduser(f"~/Dir/projects/IPLVE/data/outputs/LM_out_reduced")
        embeddings_reduced_dir = os.path.expanduser(f"~/Dir/projects/IPLVE/data/embeddings/LM_emb_reduced")

        if not os.path.exists(output_reduced_dir):
            os.makedirs(output_reduced_dir)
        if not os.path.exists(embeddings_reduced_dir):
            os.makedirs(embeddings_reduced_dir)

        reduced_encodings_path = f"{output_reduced_dir}/{args.model_name}_encodings_reduced_{args.n_components}.npy"
        embeddings_path = f"{embeddings_reduced_dir}/{args.model_name}_reduced_{args.n_components}"
        
        logger.info('Reducing dimensionality')
        final_target = reduce_encoding_size(targets, reduced_encodings_path, args.n_components)
        logger.info('Reducing dimensionality is completed!')
    else:
        final_target = targets
    
    format_embeddings(final_target, words_in_sentences, embeddings_path)
    
    # if targets.shape[1] > 2048:
    #     reduced_targets = reduce_encoding_size(targets, f'./data/outputs_opt/reduced_{args.model_name}_{args.n_components}.npy', args.n_components)
    #     format_embeddings(reduced_targets, words_in_sentences, f'./data/embeddings_opt/reduced_{args.model_name}_{args.n_components}')
    # else:
    #     format_embeddings(targets, words_in_sentences, embeddings_path)
    logger.info('==========Format complete==========')
