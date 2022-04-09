from ast import parse
from multiprocessing import pool
from tqdm import tqdm
import numpy as np
import argparse

import torch
torch.manual_seed(100)
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW

from utils.metrics import f1_score
from utils.char_label import word_to_char_level_label
from utils.ner_dataset import get_dataset
from utils.map_offset_to_word import offset_to_word
from model.bert import BertNER

label2id =  {"O": 0, "I": 1}
id2label = {_id: _label for _label, _id in list(label2id.items())}


def train_epoch(device, train_loader, model, optimizer, scheduler, pooling, epoch):
    """
    trainer for each epoch
    """
    model.train()
    train_losses = 0

    for _, batch_samples in enumerate(tqdm(train_loader)):
        batch_data, batch_flair_input, batch_masks, batch_labels, sentence, _, _, length, offset = batch_samples

        # shift tensors to GPU if available
        batch_data = batch_data.to(device)
        batch_masks = batch_masks.to(device)
        batch_labels = batch_labels.to(device)
        batch_flair_input = batch_flair_input.to(device)

        loss = model(batch_data,
                    flair_input=batch_flair_input,
                    attention_mask=batch_masks, 
                    labels=batch_labels,
                    sentence=sentence,
                    offset=offset,
                    length=length,
                    method=pooling)
        loss = loss[0]
        train_losses += loss.item()
        model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5)
        optimizer.step()
        scheduler.step()

    train_loss = round(float(train_losses) / len(train_loader), 2)
    print("Epoch: {}, train loss: {}".format(epoch, train_loss))


def eval_epoch(device, data_loader, model, pooling, epoch=1, mode="dev"):
    """
    used for evaluate the dev set or test set
    """
    model.eval()
    pred_labels = []
    true_labels = []

    losses = 0
    for _, batch_samples in enumerate(tqdm(data_loader)):
        pred_tags = []

        batch_data, batch_flair_input, batch_masks, batch_labels, sentences, origional_sentences, origional_labels, length, offset = batch_samples
        # shift tensors to GPU if available
        batch_data = batch_data.to(device)
        batch_masks = batch_masks.to(device)
        batch_labels = batch_labels.to(device)
        batch_flair_input = batch_flair_input.to(device)

        batch_output = model(batch_data,
                    flair_input=batch_flair_input,
                    attention_mask=batch_masks, 
                    labels=batch_labels,
                    sentence=sentences,
                    offset=offset,
                    length=length,
                    method=pooling)

        losses += batch_output[0].item()
        batch_output= batch_output[1].detach().cpu().numpy()
        batch_labels = batch_labels.to('cpu').numpy()
        batch_labels = batch_labels[:, :batch_output.shape[1]]
        pred_tags.extend([[id2label.get(idx.item()) for idx in indices] for indices in np.argmax(batch_output, axis=2)])

        # get character-level label from word-level tags
        for i in range(len(length)):
            data_tobe_convert = {}
            data_tobe_convert["mapping"] = set(offset_to_word(offset[i], length[i], sentences[i]))
            data_tobe_convert["tags"] = pred_tags[i]
            data_tobe_convert["sentence"] = origional_sentences[i]
            data_tobe_convert["words"] = sentences[i]
            pred_labels.append(word_to_char_level_label(data_tobe_convert))
            true_labels.append(origional_labels[i])

    scores = round(f1_score(pred_labels, true_labels), 4)
    losses = round(float(losses) / len(data_loader), 2)
    if mode == "dev":
        print("Epoch: {}, val f1: {}, val loss: {}".format(epoch, scores * 100, losses))
    else:
         print("Test f1: {}, test loss: {}".format(scores * 100, losses))


def train_model(device, train_loader, dev_loader, model, optimizer, scheduler, epoch_num, pooling):
    """
    used for training bert model
    """
    for epoch in range(1, epoch_num + 1):
        train_epoch(device, train_loader, model, optimizer, scheduler, pooling, epoch)
        eval_epoch(device, dev_loader, model, pooling, epoch=epoch, mode="dev")

    print("Training Finished!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True, help="base path to load data")
    parser.add_argument("--flair_model", type=str, required=True, help="base path to load data")
    parser.add_argument("--pooling", type=str, default="sum", help="pooling method to aggregate subword embedding, sum or mean")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--num_epoch", type=int, default=10, help="number of training epochs")
    parser.add_argument("--early_stopping", type=int, default=3, help="stop at which epoch")
    args = parser.parse_args()

    training_set = get_dataset("%s/tsd_train.csv" % args.base_path, split="train", flair_model_path=args.flair_model)
    trial_set = get_dataset("%s/tsd_trial.csv" % args.base_path, split="trial", flair_model_path=args.flair_model)
    test_set = get_dataset("%s/tsd_test.csv" % args.base_path, split="test", flair_model_path=args.flair_model)

    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, collate_fn=trial_set.collate_fn)
    trial_loader = DataLoader(trial_set, batch_size=args.batch_size, shuffle=False, collate_fn=trial_set.collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=trial_set.collate_fn)

    torch.set_num_threads(4)
    model = BertNER.from_pretrained("bert-base-uncased")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    bert_optimizer = list(model.bert.named_parameters())
    classifier_optimizer = list(model.classifier.named_parameters())
    crf_optimizer = list(model.crf.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
        'lr': 1e-5, 'weight_decay': 0.01},
        {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
        'lr': 1e-5, 'weight_decay': 0.0},
        {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
        'lr': 1e-4, 'weight_decay': 0.01},
        {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
        'lr': 1e-4, 'weight_decay': 0.0},
        {'params': model.crf.parameters(), 'lr': 5e-4}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4, correct_bias=False)
    train_steps_per_epoch = len(training_set) // args.batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=train_steps_per_epoch,
                                                num_training_steps=args.num_epoch * train_steps_per_epoch)
    
    print("--------Start Training!--------")
    train_model(device, train_loader, trial_loader, model, optimizer, scheduler, 
                args.early_stopping, args.pooling)

    print("--------Start Testing!--------")
    eval_epoch(device, test_loader, model, args.pooling, mode="test")