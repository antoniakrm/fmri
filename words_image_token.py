# def format_embeddings(X, words, embeddings_path):

#   with open(f"{embeddings_path}.txt", 'w') as outfile:
#     outfile.write(f'{str(X.shape[0])} {str(X.shape[1])}\n')
#     for word, vec in zip(words, X):
#       outfile.write(
#           f"{word.strip().lower()} {' '.join([str(v) for v in vec.tolist()])}\n")
#     outfile.close()

def format_train_eval(bert_all_path, image_classes_path, training, n):
    berts = open(bert_all_path).readlines()
    bert_words = [i.strip('\n').split(' ', 1)[0] for i in berts]
    bert_vects = [i.strip('\n').split(' ', 1)[1] for i in berts]
    image_classes_words = [i.strip('\n').split(': ', 1)[1].lower().replace(' ','_') for i in open(image_classes_path).readlines()]
    images_size = len(image_classes_words)
    with open(f'{bert_all_path[:-4]}_{training}.txt', 'w') as biw:
        if training == 'train':
            biw.write(f'{str(round(images_size*0.7)-1)} {str(n)}\n')
            for iword in image_classes_words[:round(images_size*0.7)-1]:
                # if iword in bert_words:
                biw.write(berts[bert_words.index(iword)])
            print(round(images_size*0.7)-1)
        else:
            biw.write(f'{str(images_size-round(images_size*0.7)+1)} {str(n)}\n')
            for iword in image_classes_words[round(images_size*0.7)-1:]:
                # if iword in bert_words:
                biw.write(berts[bert_words.index(iword)])
            print(images_size-round(images_size*0.7)+1)
        biw.close()
    # if training != 'train':
    #     list_size = X.shape[0] - round(X.shape[0]*0.7)
    #     words = words[round(X.shape[0]*0.7):]
    #     X = X[round(X.shape[0]*0.7):]
    # else:
    #     list_size = round(X.shape[0]*0.7)
    #     X = X[:list_size]
    #     words = words[:list_size]
    # with open(f"{embeddings_path}_{training}.txt", 'w') as outfile:
    #     outfile.write(f'{str(list_size)} {str(X.shape[1])}\n')
    #     for word, vec in zip(words, X):
    #         outfile.write(
    #             f"{word.strip().lower()} {' '.join([str(v) for v in vec.tolist()])}\n"
    #         )
    #     outfile.close()

if __name__ == "__main__":
    bert_all_path = './data/embeddings/bert-mini.txt'
    image_classes_path = './data/image_id_over100.txt'
    embeddings_path = './data/embeddings'
    n = 256
    # format_train_eval(bert_all_path, image_classes_path, 'train', n)
    # format_train_eval(bert_all_path, image_classes_path, 'eval', n)