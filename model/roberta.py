# @title  { form-width: "1px" }
import torch
import torch.nn as nn
from transformers import RobertaModel
from transformers.modeling_bert import BertPreTrainedModel
from torch.nn.utils.rnn import pad_sequence

from model.crf import CRF


class RobertaNER(BertPreTrainedModel):
    def __init__(self, config):
        super(RobertaNER, self).__init__(config)
        self.num_labels = 2
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(2304, 1152, batch_first=True, bidirectional=True, num_layers=2)
        self.classifier = nn.Linear(2304, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, batch_len=None):
        # input_ids, input_token_starts = input_data
        outputs = self.roberta(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               output_hidden_states=True)

        sequence_output = outputs[0]
        hidden_layer1 = outputs[2][1]
        hidden_layer6 = outputs[2][6]
        hidden_layer4 = outputs[2][4]
        hidden_layer8 = outputs[2][8]

        origin_sequence_output = [layer[1:] for layer in sequence_output]
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)

        # concat 1 6 12 layers
        origin_hidden_layer1 = [layer[1:] for layer in hidden_layer1]
        padded_hidden_layer1 = pad_sequence(origin_hidden_layer1, batch_first=True)
        origin_hidden_layer6 = [layer[1:] for layer in hidden_layer6]
        padded_hidden_layer6 = pad_sequence(origin_hidden_layer6, batch_first=True)
        padded_sequence_output = torch.cat([padded_hidden_layer1, padded_hidden_layer6, padded_sequence_output], dim=2)
        # concat 4 8 12 layers
        # origin_hidden_layer4 = [layer[1:] for layer in hidden_layer4]
        # padded_hidden_layer4 = pad_sequence(origin_hidden_layer4, batch_first=True)
        # origin_hidden_layer8 = [layer[1:] for layer in hidden_layer8]
        # padded_hidden_layer8 = pad_sequence(origin_hidden_layer8, batch_first=True)
        # padded_sequence_output = torch.cat([padded_hidden_layer4, padded_hidden_layer8, padded_sequence_output], dim=2)
        # padded_sequence_output: torch.Size([32, 95, 2304])

        dropout_output = self.dropout(padded_sequence_output)
        lstm_output, _ = self.lstm(dropout_output)
        logits = self.classifier(lstm_output)
        outputs = (logits,)

        if labels is not None:
            origional_labels = [layer[1:] for layer in labels]
            for i in range(len(origional_labels)):
                zero = torch.zeros_like(origional_labels[i])
                origional_labels[i] = torch.where(origional_labels[i] > -1, origional_labels[i], zero)
            padded_labels = pad_sequence(origional_labels, batch_first=True)
            loss_mask = padded_labels.gt(-100)
            loss = self.crf(logits, padded_labels, mask=loss_mask) * (-1)
            outputs = (loss,) + outputs

        return outputs
