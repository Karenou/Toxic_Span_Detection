"""
the file saves the fasttext embedding for train, trial and test set first
to save memory usage and IO time in later model training 
"""
import sys
sys.path.append('/home/juneshi/Documents/Toxic_Span_Detection')

import torch
from gensim.models import KeyedVectors
from flair.embeddings import BytePairEmbeddings
from flair.data import Sentence

import numpy as np
import argparse
import h5py

from utils.ner_dataset import get_dataset


def load_word2vec(model_path):
    """
    @param model_path: path to load pretrained word2vec model, embed_dim=300
    """
    model = KeyedVectors.load_word2vec_format(model_path, binary=False)
    return model


def get_oov_embedding(token, model, embed_dim):
    """
    @param tokens: one token
    @param model: language model
    @param embed_dim: word vector embedding dimension
    return a np array (1, embed_dim)
    """
    sentence = Sentence(token)
    model.embed(sentence)

    for token in sentence:
        # the returned embedding is 2 * embed_dim, repeat twice, keep the first half only
        embedding = torch.unsqueeze(token.embedding[:embed_dim], 0)

    embedding = embedding.cpu().detach().numpy()
    return embedding


def get_embedding(data_path, save_path, ft_model, lm_model, split="train", embed_dim=300):
    """
    @param data_path: base path to load toxic data
    @param save_path: path to save the h5 file
    @param ft_model: fast text word vectors
    @param lm_model: flair byte pair embedding language model
    @param embed_dim: embedding dimension
    write embeddings to h5 file
    """
    dataset = get_dataset("%s/tsd_%s.csv" % (data_path, split), split=split, flair_model_path=None)
    print("Load %s dataset" % split)
    print("There are %d document" % len(dataset))

    output_path = "%s/%s_ft_embedding.h5" % (save_path, split)
    embed_hdf5 = h5py.File(output_path, 'w')

    for i in range(len(dataset)):
        embeddings = []

        sentence = dataset.sentences[i]
        for token in sentence:
            if token in ft_model:
                embedding = np.expand_dims(ft_model[token], axis=0)
            else:
                embedding = get_oov_embedding(token, lm_model, embed_dim)
            embeddings.append(embedding)

        embeddings = np.concatenate(embeddings)

        # write to h5
        embed_hdf5[str(i)] = embeddings

    print("Finish writing to %s" % (output_path))
    embed_hdf5.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="base path to load toxic data")
    parser.add_argument("--model_path", type=str, required=True, help="base path to load fasttext word vector")
    parser.add_argument("--save_path", type=str, required=True, help="path to save embeddings")
    parser.add_argument("--embed_dim", type=int, default=300, help="word vector embedding dimension")
    args = parser.parse_args()

    # load fasttext word vectors
    print("Loading fast text word vectors")
    ft_model = load_word2vec(args.model_path)

    # init oov embedding, choose the embed_di=300
    print("Loading falir BytePairEmedding model")
    lm_model = BytePairEmbeddings('en', dim=args.embed_dim)

    # save train embedding
    get_embedding(args.data_path, args.save_path, ft_model, lm_model, split="train", embed_dim=args.embed_dim)

    # save trial embedding
    get_embedding(args.data_path, args.save_path, ft_model, lm_model, split="trial", embed_dim=args.embed_dim)

    # save test embedding
    get_embedding(args.data_path, args.save_path, ft_model, lm_model, split="test", embed_dim=args.embed_dim)


