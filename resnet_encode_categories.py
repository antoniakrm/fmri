import os
import torch
import numpy as np
from img2vec_pytorch import Img2Vec
from encode_utils import *
from tqdm import tqdm
from PIL import Image

# from sklearn.decomposition import PCA

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
  embeddings_path = f"{args.emb_dir}/{args.model_name}_{args.n_components}"

  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  logger.info('Encoding images')
  targets, image_classes = resnet_encode(model, os.path.expanduser(args.image_dir), encodings_path, image_classes_path, args.image_classes_id)
  logger.info('Encoding is completed!')

  if args.n_components < targets.shape[1]:
      output_reduced_dir = os.path.expanduser(f"~/Dir/projects/IPLVE/data/outputs/res_out_reduced")
      embeddings_reduced_dir = os.path.expanduser(f"~/Dir/projects/IPLVE/data/embeddings/res_emb_reduced")

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
    
  format_embeddings(final_target, image_classes, embeddings_path)
    
    # if targets.shape[1] > 2048:
    #     reduced_targets = reduce_encoding_size(targets, f'./data/outputs_opt/reduced_{args.model_name}_{args.n_components}.npy', args.n_components)
    #     format_embeddings(reduced_targets, words_in_sentences, f'./data/embeddings_opt/reduced_{args.model_name}_{args.n_components}')
    # else:
    #     format_embeddings(targets, words_in_sentences, embeddings_path)
  logger.info('==========Format complete==========')
  # if args.n_components != encoded_image_classes.shape[1]:
  #   logger.info('Reducing dimensionality')
  #   reduced_image_classes = reduce_encoding_size(encoded_image_classes, reduced_encodings_path, args.n_components)
  #   logger.info('Reducing dimensionality is completed!')
  # else:
  #   reduced_image_classes = encoded_image_classes

  # logger.info('Formatting embeddings')
  # if not os.path.exists(args.emb_dir):
  #   os.makedirs(args.emb_dir)
  # embeddings_path = f"{args.emb_dir}/{args.model_name}_{reduced_image_classes.shape[1]}.txt"
  # format_embeddings(reduced_image_classes, image_classes, embeddings_path)
  # logger.info('Format is completed!')

  # logger.info('Reducing dimensionality')
  # reduced_image_classes = reduce_encoding_size(encoded_image_classes, reduced_encodings_path, args.n_components)
  # logger.info('Formatting embeddings')
  # format_embeddings(reduced_image_classes, image_classes, embeddings_path)


if __name__ == "__main__":
    main()  