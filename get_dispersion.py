import numpy as np
import time
from itertools import combinations
import os
from scipy.spatial.distance import cosine
# import numba as nb
import math
import multiprocessing as mp

# images_embeddigns_path = os.path.expanduser('~/Dir/projects/IPLVE/data/embeddings/images_embeddings_temp/n00005787.txt')

def get_dispersion(root_path, images_embeddigns_path):
    file_path = os.path.join(root_path, images_embeddigns_path)
    images_embeddings = open(file_path).readlines()
    embeddings_nums, vec_size = images_embeddings[0].split()
    name = images_embeddings[1].rstrip().split(' ', 1)[0][:-2]
    # cos_nn = torch.nn.CosineSimilarity(dim=1)
    vectors = []
    for i, line in enumerate(images_embeddings[1:]):
        _, vect = line.rstrip().split(' ', 1)
        vect = np.fromstring(vect, sep=' ')
        # print(vect.shape)
        vectors.append(vect)

    relations = list(combinations(vectors, 2))
    # print(len(relations))
    cos_results = []
    for src, tgt in relations:
        # count = 0
        # if (src != tgt).any():
        cos_dis1 = cosine(src, tgt)
        cos_results.append(cos_dis1)
            # count += 1
    # print(len(cos_results))
            # cos_dis2 = cos_nn(torch.FloatTensor(src).unsqueeze(0), torch.FloatTensor(tgt).unsqueeze(0))
            # print(cos_dis1, cos_dis2)

    cos_avg = np.mean(cos_results)
    # print(cos_avg)
    return cos_avg, name

def get_dispersion_multi(root_path, image_categories):
    categories_dis = {}
    for image_category in image_categories:
        dis, name = get_dispersion(root_path, image_category)
        categories_dis[name] = dis
    return categories_dis

def main():
    # root_path = os.path.expanduser("~/Dir/projects/IPLVE/data/embeddings/seg_images_embeddings")
    root_path = os.path.expanduser("~/Dir/projects/IPLVE/data/embeddings/res_images_embeddings")
    # root_path = os.path.expanduser("~/Dir/projects/IPLVE/data/embeddings/embeddings_test")
    categories = os.listdir(root_path)
    num_categories = len(categories)
    time_start= time.time()
    num_cpus = 8
    block_size = math.ceil(num_categories / num_cpus)
    p = mp.Pool(processes=num_cpus)
    
    res = [p.apply_async(func=get_dispersion_multi, args=(root_path, categories[i*block_size:i*block_size+block_size])) for i in range(num_cpus)]

    cos_res_list = [i.get() for i in res]
    categories_dis_dict = {}
    for i in cos_res_list:
        categories_dis_dict.update(i)
    categories_dis_sorted = sorted(categories_dis_dict.items(), key = lambda kv:(kv[1], kv[0]))
    with open('./data/sorted_dispersion_res.txt','w') as ssw:
        for i in categories_dis_sorted:
            ssw.write(f"{i[0]}: {i[1]}\n")
        ssw.close()
    time_end = time.time()
    # print(categories_dis)
    print('time cost', time_end - time_start, 's')
    
if __name__ == "__main__":
    main()  
