# Introduction

This repository contains code for exploring isomorphism between pre-trained language and vision embedding spaces. Implement transformer-based language models ([BERT](https://arxiv.org/abs/1810.04805), [GPT2](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf), [OPT](https://arxiv.org/abs/2205.01068)) to get words embeddings, and implemennt [ResNet](https://arxiv.org/abs/1512.03385) and [Segformer](https://arxiv.org/abs/2105.15203) to get images embeddings.

Before running the code, please download fasttext identification [model](https://fasttext.cc/docs/en/language-identification.html)

File path | Description
'''
IPLVEs
├── configs
│   ├── bert_config.yaml
│   ├── gpt2_config.yaml
│   ├── opt_config.yaml
│   ├── resnet_config.yaml
│   └── segformer_config.yaml
├── data
│   └── wordlist.txt
├── env.yaml
├── __init__.py
├── LICENSE
├── main.py
├── MUSE
│   └── ...
├── pretrained
│   └── lid.176.bin
├── README.md
└── src
    ├── bert_word_emb.py
    ├── gpt2_word_emb.py
    ├── __init__.py
    ├── opt_word_emb.py
    ├── resnet_encode_categories.py
    ├── Segformer_encode.py
    └── utils
        ├── build_dico_shuffle.py
        ├── build_dispersion_dictionaries.py
        ├── encode_util.py
        ├── get_dispersion.py
        ├── get_frequency.py
        ├── get_polysemy.py
        ├── __init__.py
        ├── io_util.py
        └── sentences_downloader.py
'''

## Get word embeddings

To get the embeddings of specific words in the wordlist, simply run:
```bash
python main.py --config ./configs/bert_config.yaml
```
## Get image embeddings
To get the embeddings of specific image class, simply run:
```bash
python main.py --config ./configs/segformer_config.yaml
```
## Align word and image embeddings
To learn a mapping between the source and the target space, simply run:
```bash
python MUSE/supervised.py  --src_lang image --tgt_lang word --emb_dim=512 --seed 42 --dico_train train_dict_path --dico_eval eval_dict_path --src_emb source_emb_path --tgt_emb target_emb_path --normalize_embeddings center --n_refinement 0;  
```