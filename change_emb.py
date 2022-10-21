import os

def change_emb(root_dir):
    ids_names_pairs = open('./data/image_ids_wiki_using.txt').readlines()
    for dir in os.listdir(root_dir):
        embeddings_res = open(os.path.join(root_dir, dir)).readlines()

        ids = [i.split(': ')[0] for i in ids_names_pairs]
        vecs = [i.strip().split(' ',1)[1] for i in embeddings_res[1:]]
        _, size = embeddings_res[0].split(' ')


        idx = []
        for i in ids:
            idx.append(ids.index(i))
        idx = list(dict.fromkeys(idx))

        vect = [vecs[i] for i in idx]
        ids_unique = [ids[i] for i in idx]
        res = [f'{i} {j}' for i,j in zip(ids_unique, vect)]
        # final_res = list(dict.fromkeys(res))

        ids = [i.split(': ')[0] for i in ids_names_pairs]
        vecs = [i.strip().split(' ',1)[1] for i in embeddings_res[1:]]

        with open(f'./data/embeddings/embeddings_seg/nvidia/{dir}', 'w') as ec:
            ec.write(f'{len(res)} {size}')    
            ec.write('\n'.join(res))


if __name__ == "__main__":
    root_dir = './data/embeddings/embeddings_seg_old/nvidia'
    change_emb(root_dir)
    