
from transformers import SegformerFeatureExtractor, SegformerModel
from PIL import Image
from encode_utils import *
from tqdm import tqdm
import os
import torch
import numpy as np
from torch.utils.data import Dataset

torch.cuda.empty_cache()
# from sklearn.decomposition import PCA

class ImageDataset(Dataset):
    def __init__(self, image_dir, image_category_id, extractor, resolution) -> None:
        super(ImageDataset, self).__init__()
        self.image_root = image_dir
        self.id_name_pairs = open(image_category_id).readlines()
        self.ids = [i.strip('\n').split(': ')[0] for i in self.id_name_pairs]
        self.names = [i.strip('\n').split(': ')[1] for i in self.id_name_pairs]
        self.extractor = extractor
        self.MAX_SIZE = 200
        self.RESOLUTION_HEIGHT = resolution
        self.RESOLUTION_WIDTH = resolution
        self.CHANNELS = 3

    def __len__(self):
        return len(self.id_name_pairs)
    
    def __getitem__(self, index):
        images = []
        category_path = os.path.join(self.image_root, self.ids[index])
        for filename in os.listdir(category_path):
            try:
                images.append(Image.open(os.path.join(category_path, filename)).convert('RGB'))
            except:
                print('Failed to pil', filename)
        # print(len(images))

        category_size = len(images)
        values = self.extractor(images=images, return_tensors="pt")
        inputs = torch.zeros(self.MAX_SIZE, self.CHANNELS, self.RESOLUTION_HEIGHT, self.RESOLUTION_WIDTH)
        with torch.no_grad():
            inputs[:category_size,:,:,:].copy_(values.pixel_values)
        
        return inputs, (self.names[index], category_size)
        
device = "cuda" if torch.cuda.is_available() else "cpu"

def segformer_encode(model, dataloader, encoding_path, image_classes_path):
    if os.path.exists(encoding_path):
        print('Loading existing encoding file', encoding_path)
        encoded_image_classes = np.load(encoding_path)
        image_categories = open(image_classes_path).read().strip().lower().replace(' ','_').split('\n')
        return encoded_image_classes, image_categories
    
    model.eval()
    model = model.to(device)

    categories_encode = []
    image_categories = []
    # images_encode = []
    # images_name = []
    for inputs, (names, category_size) in tqdm(dataloader, mininterval=1800.0, maxinterval=3600.0):
        # inputs.pixel_values = inputs.pixel_values.squeeze(dim=0)
        # inputs = inputs.pixel_values.to(device)
        inputs_shape = inputs.shape
        # inputs_chunks = torch.chunk(inputs, inputs_shape[0], dim=0)
        # for idx, inputs_chip in enumerate(inputs_chunks):
        # inputs = torch.cat([inputs[i,:category_size[i],:,:,:] for i in inputs_shape[0]], dim=0)
        inputs = inputs.reshape(-1, inputs_shape[2],inputs_shape[3],inputs_shape[4]).to(device)
        # print(inputs[130,:,:,:])

        # for b5
        bs_features = []
        bs = 100
        batches = [inputs[i:i+bs] for i in range(0, len(inputs), bs)]
        for i in batches:
            with torch.no_grad():
                outputs = model(pixel_values=i)
            bs_features.append(outputs.last_hidden_state)
        bs_features = torch.cat(bs_features, axis=0).cpu()
        chunks = torch.chunk(bs_features, inputs_shape[0], dim=0)

        # for b0-4
        # with torch.no_grad():
        #     outputs = model(pixel_values=inputs)
        # chunks = torch.chunk(outputs.last_hidden_state.cpu(), inputs_shape[0], dim=0)

        for idx, chip in enumerate(chunks):
            # features for every image
            images_features = np.mean(chip[:category_size[idx]].numpy(), axis=(2,3), keepdims=True).squeeze()
            # features for categories
            
            category_feature = np.expand_dims(images_features.mean(axis=0), 0)
            # print(features.shape)
            # print(features_exp.shape)
            image_categories.append(names[idx])
            images_name = [f"{names[idx]}_{i}" for i in range(category_size[idx])]
            # images_name.append([f'{names[idx]}_{i}' for i in range(category_size[idx])])
            # images_encode.append(images_features)
            categories_encode.append(category_feature)

            # save images' features
            format_embeddings(images_features, images_name, 
            os.path.expanduser(f"~/Dir/projects/IPLVE/data/embeddings/seg_images_embedddings/{names[idx]}.txt"))

    categories_encode = np.concatenate(categories_encode)
    # images_encode = np.concatenate(images_encode)
    # images_name = np.concatenate(images_name)

    np.save(encoding_path, categories_encode)
    # np.save('/home/kfb818/Dir/projects/IPLVE/names.npy',images_name)
    # np.save('/home/kfb818/Dir/projects/IPLVE/images_seg_encode.npy', images_encode)

    with open(image_classes_path, 'w') as f:
        f.write('\n'.join(image_categories))
    
    return categories_encode, image_categories
    # return categories_encode, image_categories, images_encode, images_name

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

    feature_extractor = SegformerFeatureExtractor.from_pretrained(args.model_name, \
        cache_dir=os.path.expanduser(f"~/.cache/huggingface/transformers/models/{args.model_name}"), output_hidden_states=True)
    model = SegformerModel.from_pretrained(args.model_name, \
        cache_dir=os.path.expanduser(f"~/.cache/huggingface/transformers/models/{args.model_name}"), \
        output_hidden_states=True, return_dict=True)
    
    Resolution = int(args.model_name[-3:])
    imageset = ImageDataset(image_dir=os.path.expanduser(args.image_dir), image_category_id=args.image_classes_id, extractor=feature_extractor, resolution=Resolution)
    batch_size = 1
    image_dataloader = torch.utils.data.DataLoader(imageset, batch_size=batch_size, num_workers=8, pin_memory=True)

    logger.info('Encoding images')
    # encoded_image_classes, image_classes = segformer_encode(args.model_name, os.path.expanduser(args.image_dir), encodings_path, image_classes_list_path, args.image_classes_id)
    categories_encode, image_categories = segformer_encode(model, dataloader=image_dataloader, \
         encoding_path=encodings_path, image_classes_path=image_classes_list_path)
    logger.info('Encoding is completed!')

    # if args.n_components != encoded_image_classes.shape[1]:
    #     logger.info('Reducing dimensionality')
    #     reduced_image_classes = reduce_encoding_size(encoded_image_classes, reduced_encodings_path, args.n_components)
    #     logger.info('Reducing dimensionality is completed!')
    # else:
        # reduced_image_classes = encoded_image_classes

    logger.info('Formatting embeddings')
    if not os.path.exists(args.emb_dir):
        os.makedirs(f'{args.emb_dir}/{args.model_name}')
    # embeddings_path = f"{args.emb_dir}/{args.model_name}_{reduced_image_classes.shape[1]}.txt"
    embeddings_path = f"{args.emb_dir}/{args.model_name}_{categories_encode.shape[1]}.txt"
    # format categories
    format_embeddings(categories_encode, image_categories, embeddings_path)
    # format images
    # embeddings_path = f"{args.emb_dir}/{args.model_name}_images_{categories_encode.shape[1]}.txt"
    # format_embeddings(images_features, images_name, embeddings_path)
    logger.info('Format is completed!')

if __name__ == "__main__":
    main()