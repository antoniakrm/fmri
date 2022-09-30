from logging import getLogger
from MUSE.src.logger import create_logger
# from MUSE.src.utils import get_exp_path
import os
import configargparse
import subprocess
from datetime import datetime as dt


def initialize_exp(params):
    MAIN_DUMP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'dumped')
    # create the main dump path if it does not exist

    exp_folder = MAIN_DUMP_PATH if params.log_path == '' else params.log_path
    if not os.path.exists(exp_folder):
        subprocess.Popen("mkdir %s" % f'{exp_folder}', shell=True).wait()
    # params.log_path = exp_folder
    logger = create_logger(os.path.join(exp_folder, f'{dt.now().strftime("%Y%m%d%H%M%S")}.log'), vb=1)
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join(f'{k}: {str(v)}' for k, v in sorted(dict(vars(params)).items())))
    logger.info(f'The experiment will be stored in {exp_folder}')
    return logger

def config_parser():
    # import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--log_path", type=str, default="",
                        help='the path to save experiment logs')           
    parser.add_argument("--model_prefix", type=str, default="",
                        help='check if there is a model prefix from huggingface')
    parser.add_argument("--model_name", type=str, default='bert-tiny',
                        help='the name of words models')
    parser.add_argument("--output_dir", type=str, default='./data/encoding_output',
                        help='encoding output folder path')
    parser.add_argument("--emb_dir", type=str, default="./data/embeddings",
                        help='embeddings folder path')
    parser.add_argument("--wordlist_path", type=str, default="./data/wordlist.txt",
                        help='the whole wordlist file path')
    parser.add_argument("--sentences_path", type=str, default="./data/sentences.txt",
                        help='sentences file path')
    parser.add_argument("--download_path", type=str, default="./data/download",
                        help='the path to store the crawled sentences')
    parser.add_argument("--n_components", type=int, default=256,
                        help='for PCA n_components')
    parser.add_argument("--image_dir", type=str, default="./data/imagenet_21k_small",
                        help='images folder path')
    parser.add_argument("--image_classes_id", type=str, default="./data/image_id_part.txt",
                        help='the map of image classes and its ids')
    parser.add_argument("--ordered_words_path", type=str, default="./data/wordlist_ordered.txt",
                        help='Words in the sentences in order.')
    # parser.add_argument("--encodings file path", type)
    return parser