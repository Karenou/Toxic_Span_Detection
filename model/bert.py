#@title  { form-width: "1px" }
import torch
import torch.nn as nn
from transformers import BertModel
from transformers.modeling_bert import BertPreTrainedModel
from torch.nn.utils.rnn import pad_sequence

from model.crf import CRF
from utils.map_offset_to_word import offset_to_word, pooling

class BertNER(BertPreTrainedModel):
    def __init__(self, config):
        super(BertNER, self).__init__(config)
        self.num_labels = 2
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        # 768: raw bert, 1792: add flair only, 1068: add ft only, 2092: add both
        self.classifier = nn.Linear(1068, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, flair_input=None, ft_input=None, token_type_ids=None, attention_mask=None, 
                labels=None, sentence=None, offset=None, length=None, method="sum"):
        """
        @param input_ids: token_id
        @param flair_input: flair embeddings, shape (batch_size, max_seq_len, flair_embed_dim)
        @param ft_input: fast text word embeddings, shape (batch_size, max_seq_len, ft_embed_dim)
        @param token_type_ids: ner type id
        @param attention_mask: 
        @param labels: if add flair, use word-level label; otherwise use subword-level label
        @param sentence: list of word tokens in str
        @param offset: offset mapping of subword in word in the batch
        @param length: number of subword tokens in the batch
        @param method: method to aggregate subword embeddings
        """
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

        # shape: (batch_size, max_seq_len, bert_embed_dim)
        # the max_seq_len is based on BertTokenizer (contain subwords)
        sequence_output = outputs[0]
        # drop cls token
        origin_sequence_output = [layer[1:] for layer in sequence_output]


        if flair_input is None and ft_input is None:
            padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        # otherwise add flair input or fast text input
        else:
            batch_concat_embedding = []
            word_cnt_lst = []
            for i in range(input_ids.shape[0]):
                # perform mean / sum pooling to get the bert embedding for original tokens (without subwords)
                mapping = offset_to_word(offset[i], length[i], sentence[i])
                bert_embedding = pooling(origin_sequence_output[i], mapping, method=method)

                # concat bert, flair and / or ft embedding at original token level
                n_words = bert_embedding.shape[0]
                if flair_input is not None and ft_input is not None:
                    concat_embedding = torch.concat([bert_embedding, flair_input[i][:n_words], ft_input[i][:n_words]], dim=1)
                elif flair_input is not None:
                    concat_embedding = torch.concat([bert_embedding, flair_input[i][:n_words]], dim=1)
                else:
                    concat_embedding = torch.concat([bert_embedding, ft_input[i][:n_words]], dim=1)
                batch_concat_embedding.append(concat_embedding)
                
                word_cnt_lst.append(len(set(mapping)))
            # pad to same length
            padded_sequence_output = pad_sequence(batch_concat_embedding, batch_first=True)
            
        dropout_output = self.dropout(padded_sequence_output)
        logits = self.classifier(dropout_output)
        outputs = (logits,)

        if labels is not None:
            # if add flair or fasttext input, use word-level label
            if flair_input is not None or ft_input is not None:
                origional_labels = [layer[:n] for layer, n in zip(labels, word_cnt_lst)]
            # otherwise, use subword-level label
            else:
                origional_labels = [layer[1:] for layer in labels]

            for i in range(len(origional_labels)):
                zero = torch.zeros_like(origional_labels[i])
                origional_labels[i] = torch.where(origional_labels[i]>-1, origional_labels[i], zero)
            padded_labels = pad_sequence(origional_labels, batch_first=True)
            loss_mask = padded_labels.gt(-100)
            
            loss = self.crf(logits, padded_labels, mask=loss_mask) * (-1)
            outputs = (loss,) + outputs

        return outputs

class BertLstmNER(BertPreTrainedModel):
    def __init__(self,config):
        super(BertLstmNER, self).__init__(config)
        self.num_labels = 2
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.5)
        # raw bert: (768, 384)
        # add flair: (1792, 1152)
        # add ft: (1068, 384)
        # add both: (2092, ?)
        self.input_dim = 1792
        self.lstm_hidden_dim = 1152
        self.lstm = nn.LSTM(self.input_dim, self.lstm_hidden_dim, batch_first=True, bidirectional=True, num_layers = 2)
        self.classifier = nn.Linear(self.lstm_hidden_dim * 2, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, flair_input=None, ft_input=None, token_type_ids=None, attention_mask=None, 
                labels=None, sentence=None, offset=None, length=None, method="sum"):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        # shape (batch_size, max_seq_len, 768)
        sequence_output = outputs[0]
        origin_sequence_output = [layer[1:] for layer in sequence_output]

        if flair_input is None and ft_input is None:
            padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        # otherwise add flair input or fast text input
        else:
            batch_concat_embedding = []
            word_cnt_lst = []
            for i in range(input_ids.shape[0]):
                # perform mean / sum pooling to get the bert embedding for original tokens (without subwords)
                mapping = offset_to_word(offset[i], length[i], sentence[i])
                bert_embedding = pooling(origin_sequence_output[i], mapping, method=method)

                # concat bert, flair and / or ft embedding at original token level
                n_words = bert_embedding.shape[0]
                if flair_input is not None and ft_input is not None:
                    concat_embedding = torch.concat([bert_embedding, flair_input[i][:n_words], ft_input[i][:n_words]], dim=1)
                elif flair_input is not None:
                    concat_embedding = torch.concat([bert_embedding, flair_input[i][:n_words]], dim=1)
                else:
                    concat_embedding = torch.concat([bert_embedding, ft_input[i][:n_words]], dim=1)
                batch_concat_embedding.append(concat_embedding)
                
                word_cnt_lst.append(len(set(mapping)))
            # pad to same length
            padded_sequence_output = pad_sequence(batch_concat_embedding, batch_first=True)

        # padded_sequence_output.shape = (batch_size, max_n_word, 1792)
        # lstm_out.shape = (batch_size, max_n_word, 384 * 2)
        lstm_out, hidden = self.lstm(padded_sequence_output)
        dropout_output = self.dropout(lstm_out)
        logits = self.classifier(dropout_output)
        outputs = (logits,)

        if labels is not None:
            # if add flair or fasttext input, use word-level label
            if flair_input is not None or ft_input is not None:
                origional_labels = [layer[:n] for layer, n in zip(labels, word_cnt_lst)]
            # otherwise, use subword-level label
            else:
                origional_labels = [layer[1:] for layer in labels]

            for i in range(len(origional_labels)):
                zero = torch.zeros_like(origional_labels[i])
                origional_labels[i] = torch.where(origional_labels[i]>-1, origional_labels[i], zero)
            padded_labels = pad_sequence(origional_labels, batch_first=True)
            loss_mask = padded_labels.gt(-100)
            loss = self.crf(logits, padded_labels, mask=loss_mask) * (-1)
            outputs = (loss,) + outputs

        return outputs

class BertAvgNER(BertPreTrainedModel):
    def __init__(self,config):
        super(BertAvgNER, self).__init__(config)
        self.num_labels = 2
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.5)
        # raw bert: (768 * 3 = 2304, 1152)
        # add flair: (3328, 1664)
        # add ft: (2604, 1216)
        # add both: (3628, 1792)
        self.input_dim = 3628
        self.lstm_hidden_dim = 1792
        self.lstm = nn.LSTM(self.input_dim, self.lstm_hidden_dim, batch_first=True, bidirectional=True, num_layers = 2)
        self.classifier = nn.Linear(self.lstm_hidden_dim * 2, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, flair_input=None, ft_input=None, token_type_ids=None, attention_mask=None, 
                labels=None, sentence=None, offset=None, length=None, method="sum"):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            output_hidden_states=True)
    
        hidden_states = outputs[2]
        sequence_output = torch.cat((hidden_states[3], hidden_states[7], hidden_states[11]), 2)
        origin_sequence_output = [layer[1:] for layer in sequence_output]
        
        if flair_input is None and ft_input is None:
            padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        # otherwise add flair input or fast text input
        else:
            batch_concat_embedding = []
            word_cnt_lst = []
            for i in range(input_ids.shape[0]):
                # perform mean / sum pooling to get the bert embedding for original tokens (without subwords)
                mapping = offset_to_word(offset[i], length[i], sentence[i])
                bert_embedding = pooling(origin_sequence_output[i], mapping, method=method)

                # concat bert, flair and / or ft embedding at original token level
                n_words = bert_embedding.shape[0]
                if flair_input is not None and ft_input is not None:
                    concat_embedding = torch.concat([bert_embedding, flair_input[i][:n_words], ft_input[i][:n_words]], dim=1)
                elif flair_input is not None:
                    concat_embedding = torch.concat([bert_embedding, flair_input[i][:n_words]], dim=1)
                else:
                    concat_embedding = torch.concat([bert_embedding, ft_input[i][:n_words]], dim=1)
                batch_concat_embedding.append(concat_embedding)
                
                word_cnt_lst.append(len(set(mapping)))
            # pad to same length
            padded_sequence_output = pad_sequence(batch_concat_embedding, batch_first=True)

        lstm_out, hidden = self.lstm(padded_sequence_output)
        dropout_output = self.dropout(lstm_out)
        logits = self.classifier(dropout_output)
        outputs = (logits,)

        if labels is not None:
            # if add flair or fasttext input, use word-level label
            if flair_input is not None or ft_input is not None:
                origional_labels = [layer[:n] for layer, n in zip(labels, word_cnt_lst)]
            # otherwise, use subword-level label
            else:
                origional_labels = [layer[1:] for layer in labels]

            for i in range(len(origional_labels)):
                zero = torch.zeros_like(origional_labels[i])
                origional_labels[i] = torch.where(origional_labels[i]>-1, origional_labels[i], zero)
            padded_labels = pad_sequence(origional_labels, batch_first=True)
            loss_mask = padded_labels.gt(-100)
            loss = self.crf(logits, padded_labels, mask=loss_mask) * (-1)
            outputs = (loss,) + outputs

        return outputs