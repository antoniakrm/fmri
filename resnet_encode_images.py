import os
import io
import torch
import numpy as np
from img2vec_pytorch import Img2Vec
from encode_utils import *
from tqdm import tqdm
import math
from PIL import Image

from sklearn.decomposition import PCA

device = "cuda" if torch.cuda.is_available() else "cpu"

def resnet_encode(model_name, image_dir, encodings_path, image_classes_path, image_id_words_path=None):

  if os.path.exists(encodings_path):
    print('Loading existing encodings file', encodings_path)
    encoded_image_classes = np.load(encodings_path)
    print(encoded_image_classes.shape)
    image_classes = open(image_classes_path).read().strip().lower().replace(' ','_').split('\n')
    return encoded_image_classes, image_classes

  img2vec = Img2Vec(model=model_name, cuda=torch.cuda.is_available())
  # image_classes_ids = os.listdir(image_dir)
  if image_id_words_path == '':
    image_classes_ids = os.listdir(image_dir)
    image_classes = os.listdir(image_dir)
  else:
    ids_words = open(image_id_words_path).readlines()
    image_classes_ids = [i.strip('\n').split(': ')[0] for i in ids_words]
    image_classes = [i.strip('\n').split(': ')[1] for i in ids_words]

  def pil_image_class(image_class):
    images = []
    class_id = image_class
    # class_id = image_classes_ids[image_classes.index(image_class)]
    image_ids = os.listdir(os.path.join(image_dir, class_id))
    for filename in os.listdir(os.path.join(image_dir, class_id)):
      try:
        images.append(Image.open(os.path.join(image_dir, class_id, filename)).convert('RGB'))
      except:
        image_ids.remove(filename)
        print('Failed to pil', filename)
    return images

  def encode_one_class(images_, image_class):
    bs = 300
    batches = [images_[i:i+bs] for i in range(0, len(images_), bs)]
    # image_ids = os.listdir(os.path.join(image_dir, image_class_id_))
    # image_name = image_class
    images_names = [f"{image_class}_{i}" for i in range(len(images_))]
    features = []

    with torch.no_grad():
      for batch in batches:
        features.append(img2vec.get_vec(batch, tensor=True).to('cpu').numpy())
    features = np.concatenate(features).squeeze()

    format_embeddings(features, images_names, os.path.expanduser(f'~/Dir/projects/IPLVE/data/embeddings/res_images_embeddings/{image_name}.txt'))

    return np.expand_dims(features.mean(axis=0), 0)

  encoded_image_classes = []
  for image_class in tqdm(image_classes_ids, mininterval=300.0, maxinterval=600.0):
    if os.path.exists(os.path.expanduser(f'~/Dir/projects/IPLVE/data/embeddings/res_images_embeddings/{image_class}.txt')):
      continue
    else:
      images = pil_image_class(image_class)
      encoded_image_classes.append(encode_one_class(images, image_class))
    # print(encoded_image_classes.shape)
    # return 0

  encoded_image_classes = np.concatenate(encoded_image_classes)
  np.save(encodings_path, encoded_image_classes)

  with open(image_classes_path, 'w') as f:
    f.write('\n'.join(image_classes))

  return encoded_image_classes, image_classes


def reduce_encoding_size(X, reduced_encodings_path, n_components=128):

  if os.path.exists(reduced_encodings_path):
    return np.load(reduced_encodings_path)
  print(X.shape)
  pca = PCA(n_components=n_components)
  tr_data = pca.fit_transform(X)
  print(tr_data.shape)

  np.save(reduced_encodings_path, tr_data)

  return tr_data

def format_embeddings(X, image_classes, embeddings_path):

  with open(embeddings_path, 'w+') as outfile:
    outfile.write(f'{str(X.shape[0])} {str(X.shape[1])}\n')
    for img_name, vec in zip(image_classes, X):
      outfile.write(
          f"{img_name.lower().replace(' ', '_')} {' '.join([str(v) for v in vec.tolist()])}\n")

def main():
  parser = config_parser()
  args = parser.parse_args()
  logger = initialize_exp(args)
  if args.model_prefix:
      model = f'{args.model_prefix}/{args.model_name}'
  else:
      model = args.model_name

  encodings_path = f"{args.output_dir}/{args.model_name}_encodings.npy"
  reduced_encodings_path = f"{args.output_dir}/{args.model_name}_{args.n_components}_reduced_encodings_new.npy"
  image_classes_path = f'{args.output_dir}/{args.model_name}_image_classes.txt'
  embeddings_path = f"{args.emb_dir}/{args.model_name}_{args.n_components}_new.txt"

  if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)
  if not os.path.exists(args.emb_dir):
    os.mkdir(args.emb_dir)

  logger.info('Encoding images')
  encoded_image_classes, image_classes = resnet_encode(model, os.path.expanduser(args.image_dir), encodings_path, image_classes_path, args.image_classes_id)
  # logger.info('Reducing dimensionality')
  # if args.n_components <2048:
  #   final_target = reduce_encoding_size(encoded_image_classes, reduced_encodings_path, args.n_components)
  # else:
  #   final_target = encoded_image_classes
  # logger.info('Formatting embeddings')

  # format_embeddings(final_target, image_classes, embeddings_path)


#   id_list = os.listdir(os.path.expanduser('~/Dir/projects/summer_intern/data_words_images/images_embeddings_txt'))
#   with open(os.path.expanduser('~/Dir/projects/summer_intern/data_words_images/embeddings_sub/res34_all_images.txt'), 'w') as dim2048:
#     dim2048.write('177600 512\n')
#     for i in id_list:
#       images_pair = open(os.path.expanduser(f'~/Dir/projects/summer_intern/data_words_images/images_embeddings_txt/{i}')).readlines()[1:]
#       for content in images_pair:
#         dim2048.write(content)
#     dim2048.close()

  # # baseline = open('~/Dir/projects/summer_intern/data_words_images/embeddings_sub/all_images.txt').readlines()
  # baseline2 = open('~/Dir/projects/summer_intern/data_words_images/embeddings_sub/resnet152_2048_sub.txt').readlines()

  # vectors = []
  # words = []
  # with io.open('~/Dir/projects/summer_intern/data_words_images/embeddings_sub/res152_all_images.txt', 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
  #   for i, line in enumerate(f):
  #     if i < 1:
  #       continue
  #     word, vect = line.rstrip().split(' ', 1)
  #     vect = np.fromstring(vect, sep=' ', dtype='float64')
  #     vectors.append(vect[None])
  #     words.append(word)
  #   embeddings = np.concatenate(vectors, 0)
  #   print(embeddings.shape)
  #   f.close()

  # reduced_image_classes = reduce_encoding_size(embeddings, reduced_encodings_path, args.n_components)
  # logger.info('reduce completed')
  # format_embeddings(reduced_image_classes, words, embeddings_path)
  # logger.info('format completed')

      # image_id = [i.split(' ', 1)[0] for i in images_pair]
      # img_emb = [i.split(' ', 1)[1].split() for i in images_pair]

      # all_image_id.extend(image_id)
      # all_image_emb.extend(img_emb)

if __name__ == "__main__":
    main()  