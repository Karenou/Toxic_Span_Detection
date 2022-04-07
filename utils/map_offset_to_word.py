import torch

"""
map offset of bert subwords to original word after nltk tokenizer
pass to do mean pooling
"""

def offset_to_word(offset, length, words):
    """
    @param offset: the start and end index of the subword within corresponding word, shape(max_seq_len, 2)
    @param length: number of subwords including the cls tokens
    @param words: list of word tokens
    """
    map_lst = []
    index = 0

    # skip the first index of cls token
    for i in range(1, length):
        map_lst.append(index)

        if offset[i][1] == len(words[index]):
            index += 1

    return map_lst

def mean_pooling(embeddings, map_lst):
    """
    @param embeddings: bert embeddings, shape (max_seq_len, embed_dim)
    @param map_lst: a list to map the subword to word index
    return a tensor of embeddings at word-level (length, embed_dim)
    """
    word_embeddings = []
    tmp = [torch.unsqueeze(embeddings[0], dim=0)]

    for i in range(1, len(map_lst)+ 1):
        if i == len(map_lst) or map_lst[i] != map_lst[i-1]:
            if len(tmp) > 1:
                word_embedding = torch.concat(tmp, dim=0)
                word_embedding = torch.mean(word_embedding, dim=0, keepdims=True)
            else:
                word_embedding = tmp[0]

            word_embeddings.append(word_embedding)
            tmp = []
        
        tmp.append(torch.unsqueeze(embeddings[i], dim=0))

    word_embeddings = torch.concat(word_embeddings, dim=0)

    return word_embeddings
    