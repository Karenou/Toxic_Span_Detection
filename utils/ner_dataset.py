import itertools
import string 
import csv
import ast
import sys

import spacy
import nltk
nltk.download('punkt')

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
import numpy as np

from flair.data import Sentence
from flair.embeddings import FlairEmbeddings

SPECIAL_CHARACTERS = string.whitespace


def _contiguous_ranges(span_list):
    """Extracts continguous runs [1, 2, 3, 5, 6, 7] -> [(1,3), (5,7)]."""
    output = []
    for _, span in itertools.groupby(
        enumerate(span_list), lambda p: p[1] - p[0]):
        span = list(span)
        output.append((span[0][1], span[-1][1]))
    return output

def fix_spans(spans, text, special_characters=SPECIAL_CHARACTERS):
    """Applies minor edits to trim spans and remove singletons."""
    cleaned = []
    for begin, end in _contiguous_ranges(spans):
        while text[begin] in special_characters and begin < end:
            begin += 1
        while text[end] in special_characters and begin < end:
            end -= 1
        if end - begin > 1:
            cleaned.extend(range(begin, end + 1))
    return cleaned

def read_datafile(filename):
    """
    Reads csv file with python span list and text.
    @param filename: path of csv file
    return List[tuple(fixed_span(List[int]), text(str))]
    """
    data = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            fixed = fix_spans(ast.literal_eval(row['spans']), row['text'])
            data.append((fixed, row['text']))
    return data

def spans_to_ents(doc, spans, label):
    """
    Converts span indicies into spacy entity labels.
    @param doc: spacy document 
    @param spans: fixed spans (List[int])
    @param label: "TOXIC"
    return: a list of start and end index of toxic token, List[tuple(start_idx, end_idx + 1, "TOXIC")]
    """
    started = False
    left, right, ents = 0, 0, []
    for x in doc:
        # skip white space
        if x.pos_ == 'SPACE':
            continue
    if spans.intersection(set(range(x.idx, x.idx + len(x.text)))):
        if not started:
            left, started = x.idx, True
        right = x.idx + len(x.text)
    elif started:
        ents.append((left, right, label))
        started = False
    if started:
        ents.append((left, right, label))
    return ents

def build_ner_data_structure(data):
    """
    @param data: [(doc.text, {'entities': List[tuple(start_idx, end_idx + 1, "TOXIC")]}, text)]
    return 
        X: list of document of tokens, List[List[str]]
        y: list of list of token labels in O or I, List[List[str]]
    """
    ner_labels = []
    ner_texts = []

    for i in range(len(data)):
        skip_sample = False
        sent = data[i][2]
        words_tmp = nltk.word_tokenize(sent)
        words = []
        for j in words_tmp:
            if sent.find(j) > -1:
                words.append(j)
        word_level_labels = []
        bias = 0
        for word in words:
            index = sent.find(word)
            if index == -1:
                print("error")
                skip_sample = True
                continue

            flag = True
            for entity in data[i][1]["entities"]:
                if index + bias < entity[1] and index + bias >= entity[0]:
                    word_level_labels.append("I")
                    flag = False
                    break

            if flag:
                word_level_labels.append("O")
            bias = bias + index + len(word)
            sent = sent[index + len(word):]

        if skip_sample:
            print("ship this sample")
            continue
        ner_texts.append(words)
        ner_labels.append(word_level_labels)

    if len(ner_labels) != len(ner_texts):
        print("num of samples inconsist!")
        sys.exit()

    for i in range(len(ner_labels)):
        if len(ner_labels[i]) != len(ner_texts[i]):
            print("sentence len inconsist!")
        sys.exit()

    return ner_texts, ner_labels


def build_ner_data_structure(data):
    ner_labels = []
    ner_texts = []
    for i in range(len(data)):
        skip_sample = False
        sent_o = data[i][2]
        sent = data[i][2]
        words_tmp = nltk.word_tokenize(sent)
        words = []
        for j in words_tmp:
            if sent.find(j) > -1:
                words.append(j)
        word_level_labels = []
        bias = 0
        for word in words:
            index = sent.find(word)
            if index == -1:
                print("error")
                skip_sample = True
                continue
            flag = True
            for entity in data[i][1]["entities"]:
                if index + bias < entity[1] and index + bias >= entity[0]:
                    word_level_labels.append("I")
                    flag = False
                    break
            if flag:
                word_level_labels.append("O")
            bias = bias + index + len(word)
            sent = sent[index + len(word):]
        if skip_sample:
            print("ship this sample")
            continue
        ner_texts.append(words)
        ner_labels.append(word_level_labels)
    
    if len(ner_labels) != len(ner_texts):
        print("num of samples inconsist!")
        sys.exit()
    for i in range(len(ner_labels)):
        if len(ner_labels[i]) != len(ner_texts[i]):
            print("sentence len inconsist!")
            sys.exit()

    return ner_texts, ner_labels


class NERDataset(Dataset):
    def __init__(self, sentences, labels, max_len, origional_data, flair_model, flair_embed_dim=1024):
        """
        @param sentences: list of document in tokens
        @param labels: list of O, I labels
        @param max_len: max_seq_len in bert model
        @param original_data: list of (fixed spans, text)
        @param flair_model: path to load finetuned language model in flair
        """
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.label2id =  {"O": 0, "I": 1}
        self.id2label = {_id: _label for _label, _id in list(self.label2id.items())}
        self.sentences = sentences
        self.labels = labels
        self.origional_data = origional_data
        self.max_seq_len = max_len
        self.dataset = self.preprocess(sentences, labels, origional_data)
        self.word_pad_idx = 0
        self.label_pad_idx = -100
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.flair_model = self.load_lm(flair_model)
        self.flair_embed_dim = flair_embed_dim

    def preprocess(self, sentences, labels, origional_data):
        """
        convert to list of data
        @param sentences: a list of document in list of tokens
        @param labels: list of O, I labels
        @param original_data: a list of document in string
        """
        data = []
        for sentence, label, origional_data_item in zip(sentences, labels, origional_data):
            data.append((sentence, label, origional_data_item))
        return data

    def load_lm(self, model_path):
        """
        load flair language model
        @param model_path: path to load finetuned language model
        """
        lm_embeddings = FlairEmbeddings(model_path)
        return lm_embeddings

    def get_flair_embedding(self, sentence):
        """
        return flair embedding
        @param sentence: a list of token in string
        return np array with shape (n_tokens, flair_embed_dim)
        """
        flair_sentence = Sentence(sentence)
        self.flair_model.embed(flair_sentence)

        embeddings = []
        for token in flair_sentence:
            embedding = torch.unsqueeze(token.embedding, 0)
            embeddings.append(embedding)

        embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
        return embeddings

    def __getitem__(self, idx):
        # list of tokens
        sentence = self.dataset[idx][0]
        # list of O or I
        word_labels = self.dataset[idx][1]
        # text in string
        origional_sentence = self.dataset[idx][2][1]
        # fixed span, list of toxic position index
        origional_label = self.dataset[idx][2][0]

        encoding = self.tokenizer(sentence,
                                    is_pretokenized=True, 
                                    return_offsets_mapping=True, 
                                    padding='max_length', 
                                    truncation=True, 
                                    max_length=self.max_seq_len)
        # convert O, I to 0 and 1
        labels = [self.label2id[label] for label in word_labels] 

        length = 0
        for i in encoding["input_ids"]:
            if i != 102:
                length = length + 1
            else:
                break
        
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[0] == 0 and mapping[1] != 0:
                encoded_labels[idx] = labels[i]
                i += 1
            elif mapping[0] > 0:
                encoded_labels[idx] = labels[i-1]

        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels)
        item['sentences'] = sentence
        item['origional_sentences'] = origional_sentence
        item['origional_labels'] = origional_label
        item["tokens_len"] = length
        return item

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch):
        # encode token into int id
        input_ids = [x["input_ids"] for x in batch]
        labels = [x["labels"] for x in batch]

        # original text tokens in string after nltk tokenizer
        sentences = [x["sentences"] for x in batch]
        # length of the longest sentence in the current batch
        max_sentence_len = max([len(x) for x in sentences])

        # original text in string
        origional_sentence = [x["origional_sentences"] for x in batch]
        # fixed span of toxic labels position index
        origional_label = [x["origional_labels"] for x in batch]
        length = [x["tokens_len"] for x in batch]
        offset = [x["offset_mapping"] for x in batch]
        attention_masks = [x["attention_mask"] for x in batch]

        batch_len = len(input_ids)

        # initialize empty batch
        batch_data = self.word_pad_idx * np.ones((batch_len, self.max_seq_len))
        batch_labels = self.label_pad_idx * np.ones((batch_len, self.max_seq_len))
        batch_attention_masks = self.word_pad_idx * np.ones((batch_len, self.max_seq_len))
        batch_flair_feat = self.word_pad_idx * np.ones((batch_len, max_sentence_len, self.flair_embed_dim))

        for i in range(batch_len):
            cur_len = len(input_ids[i])
            batch_data[i][:cur_len] = input_ids[i]

            cur_tags_len = len(labels[i])
            batch_labels[i][:cur_tags_len] = labels[i]

            cur_tags_len = len(attention_masks[i])
            batch_attention_masks[i][:cur_tags_len] = attention_masks[i]
            
            # flair embedding
            curr_sentence_len = len(sentences[i])
            batch_flair_feat[i][:curr_sentence_len] = self.get_flair_embedding(sentences[i])


        # convert data to torch LongTensors
        batch_data = torch.tensor(batch_data, dtype=torch.long)
        batch_attention_masks = torch.tensor([item.detach().numpy() for item in attention_masks], dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        # convert embedding to torch FloatTensors
        batch_flair_feat = torch.tensor(batch_flair_feat, dtype=torch.float)

        return [batch_data, batch_flair_feat, batch_attention_masks, batch_labels,  sentences, origional_sentence, origional_label, length, offset]

