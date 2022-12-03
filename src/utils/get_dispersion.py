# import numba as nb
import os
from itertools import combinations

import numpy as np
from scipy.spatial.distance import cosine


def get_dispersion(root_path, images_embeddigns_path):
    file_path = os.path.join(root_path, images_embeddigns_path)
    images_embeddings = open(file_path).readlines()
    
    name = images_embeddings[1].rstrip().split(' ', 1)[0][:-2]

    vectors = []
    for _, line in enumerate(images_embeddings[1:]):
        _, vect = line.rstrip().split(' ', 1)
        vect = np.fromstring(vect, sep=' ')
        # print(vect.shape)
        vectors.append(vect)

    relations = list(combinations(vectors, 2))
    cos_results = []
    for src, tgt in relations:
        cos_dis1 = cosine(src, tgt)
        cos_results.append(cos_dis1)

    cos_avg = np.mean(cos_results)
    return cos_avg, name

def get_dispersion_multi(root_path, image_categories):
    categories_dis = {}
    for image_category in image_categories:
        
        dis, name = get_dispersion(root_path, image_category)
        categories_dis[name] = dis
    return categories_dis

