from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, SegformerModel
from PIL import Image
from modified_utils import *
from tqdm import tqdm
import os
import torch
import numpy as np

# from sklearn.decomposition import PCA

device = "cuda" if torch.cuda.is_available() else "cpu"

def segformer_encode(model_name, image_dir, encoding_path, image_classes_path, id_name_pairs=None):
    if os.path.exists(encoding_path):
        print('Loading existing encoding file', encoding_path)
        encoded_image_classes = np.load(encoding_path)
        image_classes = open(image_classes_path).read().strip().lower().replace(' ','_').split('\n')
        return encoded_image_classes, image_classes
    
    if id_name_pairs == '':
        image_classes_ids = os.listdir(image_dir)
        # ??????
        image_classes = os.listdir(image_dir)
    else:
        id_words = open(id_name_pairs).readlines()
        image_classes_ids = [i.strip('\n').split(': ')[0] for i in id_words]
        image_classes = [i.strip('\n').split(': ')[1] for i in id_words]

    def pil_image_class(image_id):
        images = []
        for filename in os.listdir(os.path.join(image_dir, image_id)):
            try:
                images.append(Image.open(os.path.join(image_dir, image_id, filename)).convert('RGB'))
            except:
                print('Failed to pil', filename)
        return images

    feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name, \
        cache_dir=os.path.expanduser(f"~/.cache/huggingface/transformers/models/{model_name}"), output_hidden_states=True)
    model = SegformerModel.from_pretrained(model_name, \
        cache_dir=os.path.expanduser(f"~/.cache/huggingface/transformers/models/{model_name}"), \
        output_hidden_states=True, return_dict=True)
    
    model.eval()
    model = model.to(device)

    def encode_one_class(images_):
        # bs = 200
        # batches = [images_[i:i+bs] for i in range(0, len(images_), bs)]
        # features = []

        # for batch in batches:
            # batch = torch.stack(batch).to(device)
        inputs = feature_extractor(images=images_, return_tensors="pt")
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # hidden_states are only from segformer encoder
        # last_encode_hidden = outputs.last_hidden_state
        # the same operation as the efficientnets in img2vec
        # features = torch.mean(last_encode_hidden, (2,3), True).numpy().squeeze()
        features = np.mean(outputs.last_hidden_state.cpu().numpy(), axis=(2,3), keepdims=True).squeeze()
        # features = np.concatenate(features).squeeze()
        # features_np_exp = np.expand_dims(features1.mean(axis=0), 0)
        # features_torch_exp = torch.unsqueeze(features_torch.mean(dim=0), 0)
        return np.expand_dims(features, 0)
        # return torch.unsqueeze(features.mean(dim=0), 0)
        # bs = 200
        # batches = [images_[i:i+bs] for i in range(0, len(images_), bs)]
        # features = []

        # for batch in batches:
        #     # batch = torch.stack(batch).to(device)
        #     inputs = feature_extractor(images=batch, return_tensors="pt")
        #     with torch.no_grad():
        #         outputs = model(**inputs.to(device))
        #     # hidden_states are only from segformer encoder
        #     # last_encode_hidden = outputs.hidden_states[-1]
        #     # the same operation as the efficientnets in img2vec
        #     features.append(torch.mean(outputs.last_hidden_state, (2,3), True))
        # features = np.concatenate(features).squeeze()
        # # features = torch.cat(features).squeeze()
        # # features_np_exp = np.expand_dims(features1.mean(axis=0), 0)
        # # features_torch_exp = torch.unsqueeze(features_torch.mean(dim=0), 0)
        # return np.expand_dims(features.mean(axis=0), 0)
        # # return torch.unsqueeze(features.mean(dim=0), 0)

    encode_image_classes = []
    for image_class in tqdm(image_classes_ids):
        unique = None
        if unique != image_class:
            images = pil_image_class(image_class)
            encode_result = encode_one_class(images)
            encode_image_classes.append(encode_result)
            unique = image_class
        else:
            encode_image_classes.append(encode_result)

    encode_image_classes = np.concatenate(encode_image_classes)
    np.save(encoding_path, encode_image_classes)

    with open(image_classes_path, 'w') as f:
        f.write('\n'.join(image_classes))
    
    return encoded_image_classes, image_classes

def format_embeddings(X, image_classes, embeddings_path):

    with open(embeddings_path, 'w') as outfile:
        outfile.write(f'{str(X.shape[0])} {str(X.shape[1])}\n')
        for img_name, vec in zip(image_classes, X):
            outfile.write(
                f"{img_name.lower().replace(' ', '_')} {' '.join([str(v) for v in vec.tolist()])}\n"
            )

def main():
    parser = config_parser()
    args = parser.parse_args()
    logger = initialize_exp(args)
    
    encodings_path = f'{args.output_dir}/{args.model_name}_encodings.npy'
    # reduced_encoding_path = f'{args.output_dir}/{args.model_name}_{args.n_components}_reduced_encodings.npy'
    image_classes_list_path = f'{args.output_dir}/{args.model_name}_image_classes.txt'
    
    if not os.path.exists(f'{args.output_dir}/{args.model_name}'):
        os.makedirs(f'{args.output_dir}/{args.model_name}')
    
    logger.info('Encoding images')
    encoded_image_classes, image_classes = segformer_encode(args.model_name, os.path.expanduser(args.image_dir), encodings_path, image_classes_list_path, args.image_classes_id)
    logger.info('Encoding is completed!')

    # if args.n_components != encoded_image_classes.shape[1]:
    #     logger.info('Reducing dimensionality')
    #     reduced_image_classes = reduce_encoding_size(encoded_image_classes, reduced_encodings_path, args.n_components)
    #     logger.info('Reducing dimensionality is completed!')
    # else:
        # reduced_image_classes = encoded_image_classes
    logger.info('Formatting embeddings')
    if not os.path.exists(args.emb_dir):
        os.makedirs(args.emb_dir)
    # embeddings_path = f"{args.emb_dir}/{args.model_name}_{reduced_image_classes.shape[1]}.txt"
    embeddings_path = f"{args.emb_dir}/{args.model_name}_{encoded_image_classes.shape[1]}.txt"
    format_embeddings(encoded_image_classes, image_classes, embeddings_path)
    logger.info('Format is completed!')

if __name__ == "__main__":
    main()