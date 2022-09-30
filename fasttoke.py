import fasttext
import fasttext.util
import numpy as np

n = 300
# fasttext.util.download_model('en', if_exists='ignore')  # English
ft = fasttext.load_model('./pretrained/cc.en.300.bin')
fasttext.util.reduce_model(ft, n)

wl = open('./data/wordlist_satisfied.txt').read().strip().lower().split('\n')
embeddings = [ft.get_sentence_vector(x.replace('_', ' ')) for x in wl]
# embeddings = []
# for phrase in wl:
#     w_p = phrase.split('_')
#     emb = [ft.get_word_vector(x) for x in w_p]
#     emb_a = np.mean(emb, axis=0)
#     embeddings.append(emb_a)

with open(f'./data/embeddings_wiki_v3/fasttext_embeddings_{n}.txt','w+') as fe:
    fe.write(f'{len(wl)} {n}\n')
    for index, embed in enumerate(embeddings):
        embeds = ' '.join([str(i) for i in embed])
        fe.write(f'{wl[index]} {embeds}\n')
    fe.close()

# def l2_norm(x):
#    return np.sqrt(np.sum(x**2))

# def div_norm(x):
#    norm_value = l2_norm(x)
#    if norm_value > 0:
#        return x * ( 1.0 / norm_value)
#    else:
#        return x

# one = ft.get_word_vector('hello')
# two = ft.get_word_vector('world')
# eos = ft.get_word_vector('\n')

# li = [one, two]

# one_two = ft.get_sentence_vector('hello world')
# one_two_avg = (div_norm(one) + div_norm(two) + div_norm(eos)) / 2
# avg = np.mean([div_norm(i) for i in li], axis=0)

# print(avg)
# is_equal = np.array_equal(one_two, one_two_avg)

# print(is_equal)


