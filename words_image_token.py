# def format_embeddings(X, words, embeddings_path):

#   with open(f"{embeddings_path}.txt", 'w') as outfile:
#     outfile.write(f'{str(X.shape[0])} {str(X.shape[1])}\n')
#     for word, vec in zip(words, X):
#       outfile.write(
#           f"{word.strip().lower()} {' '.join([str(v) for v in vec.tolist()])}\n")
#     outfile.close()


if __name__ == "__main__":
    bert_all_path = './data/embeddings/bert-mini.txt'
    image_classes_path = './data/image_id_over100.txt'
    embeddings_path = './data/embeddings'
    n = 256