
import os
import torch
import numpy as np
from img2vec_pytorch import Img2Vec
from utils import *
from tqdm import tqdm
import math
from PIL import Image

from sklearn.decomposition import PCA

device = "cuda" if torch.cuda.is_available() else "cpu"

def resnet_encode(model_name, image_dir, encodings_path, image_classes_path, image_id_words_path=None):

  if os.path.exists(encodings_path):
    print('Loading existing encodings file', encodings_path)
    encoded_image_classes = np.load(encodings_path)
    image_classes = open(image_classes_path).read().strip().lower().replace(' ','_').split('\n')
    return encoded_image_classes, image_classes
   
  # img2vec = Img2Vec(model=model_name, cuda=torch.cuda.is_available())
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
    for filename in os.listdir(os.path.join(image_dir, image_class)):
      try:
        images.append(Image.open(os.path.join(image_dir, image_class, filename)).convert('RGB'))
      except:
        print('Failed to pil', filename)
    return images
  
  img2vec = Img2Vec(model=model_name, cuda=torch.cuda.is_available())

  def encode_one_class(images_):
    bs = 200
    batches = [images_[i:i+bs] for i in range(0, len(images_), bs)]

    features = []

    with torch.no_grad():
      for batch in batches:
        features.append(img2vec.get_vec(batch, tensor=True).to('cpu').numpy())
    # print(features[0].shape)
    features = np.concatenate(features).squeeze()
    # print(features.shape)
    # print(np.expand_dims(features.mean(axis=0), 0).shape)
    return np.expand_dims(features.mean(axis=0), 0)

  encoded_image_classes = []

  for image_class in tqdm(image_classes_ids, mininterval=300.0, maxinterval=600.0):
    images = pil_image_class(image_class)
    encoded_image_classes.append(encode_one_class(images))
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
  print('Original Shape: ', X.shape)
  pca = PCA(n_components=n_components)
  tr_data = pca.fit_transform(X)
  print('Reduced Shape: ', tr_data.shape)

  np.save(reduced_encodings_path, tr_data)

  return tr_data

def format_embeddings(X, image_classes, embeddings_path):

  with open(embeddings_path, 'w') as outfile:
    outfile.write(f'{str(X.shape[0])} {str(X.shape[1])}\n')
    for img_name, vec in zip(image_classes, X):
      outfile.write(
          f"{img_name.lower().replace(' ', '_')} {' '.join([str(v) for v in vec.tolist()])}\n")

def format_train_eval(path):
    with open(path, 'r') as images_emb_reader:
        lines = images_emb_reader.readlines()
        # print(lines[0].split()[0])
        train_size = math.floor(int(lines[0].split()[0]) * 0.7)
        eval_size = int(lines[0].split()[0]) - train_size
        with open(f'{path[:-4]}_train.txt', 'w') as images_emb_train:
            images_emb_train.write(f'{train_size} {lines[0].split()[1]}\n')
            for line in lines[1:train_size+1]:
                images_emb_train.write(line)
            images_emb_train.close()
        with open(f'{path[:-4]}_eval.txt', 'w') as images_emb_eval:
            images_emb_eval.write(f'{eval_size} {lines[0].split()[1]}\n')
            for line in lines[train_size+1:]:
                images_emb_eval.write(line)
            images_emb_eval.close()
        images_emb_reader.close()

 
def main():
  parser = config_parser()
  args = parser.parse_args()
  logger = initialize_exp(args)
  if args.model_prefix:
      model = f'{args.model_prefix}/{args.model_name}'
  else:
      model = args.model_name
  
  encodings_path = f"{args.output_dir}/{args.model_name}_encodings.npy"
  reduced_encodings_path = f"{args.output_dir}/{args.model_name}_{args.n_components}_reduced_encodings.npy"
  image_classes_path = f'{args.output_dir}/{args.model_name}_image_classes.txt'

  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  logger.info('Encoding images')
  encoded_image_classes, image_classes = resnet_encode(model, os.path.expanduser(args.image_dir), encodings_path, image_classes_path, args.image_classes_id)
  logger.info('Encoding is completed!')
  if args.n_components != encoded_image_classes.shape[1]:
    logger.info('Reducing dimensionality')
    reduced_image_classes = reduce_encoding_size(encoded_image_classes, reduced_encodings_path, args.n_components)
    logger.info('Reducing dimensionality is completed!')
  else:
    reduced_image_classes = encoded_image_classes

  logger.info('Formatting embeddings')
  if not os.path.exists(args.emb_dir):
    os.makedirs(args.emb_dir)
  embeddings_path = f"{args.emb_dir}/{args.model_name}_{reduced_image_classes.shape[1]}.txt"
  format_embeddings(reduced_image_classes, image_classes, embeddings_path)
  logger.info('Format is completed!')

  # logger.info('Reducing dimensionality')
  # reduced_image_classes = reduce_encoding_size(encoded_image_classes, reduced_encodings_path, args.n_components)
  # logger.info('Formatting embeddings')
  # format_embeddings(reduced_image_classes, image_classes, embeddings_path)
  # format_train_eval(embeddings_path)

if __name__ == "__main__":
    main()  