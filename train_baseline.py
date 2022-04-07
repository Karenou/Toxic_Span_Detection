from ast import parse
from tqdm import tqdm
import numpy as np
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW

from utils.metrics import f1_score
from utils.char_label import get_char_level_label
from utils.ner_dataset import get_dataset
from model.bert import BertNER

label2id =  {"O": 0, "I": 1}
id2label = {_id: _label for _label, _id in list(label2id.items())}


def train_epoch(device, train_loader, model, optimizer, scheduler, epoch):
    """
    trainer for each epoch
    """
    model.train()
    train_losses = 0

    for _, batch_samples in enumerate(tqdm(train_loader)):
        batch_data, _, batch_masks, batch_labels, _, _, _, _, _ = batch_samples

        # shift tensors to GPU if available
        batch_data = batch_data.to(device)
        batch_masks = batch_masks.to(device)
        batch_labels = batch_labels.to(device)

        loss = model(batch_data,
                    token_type_ids=None, 
                    attention_mask=batch_masks, 
                    labels=batch_labels)
        loss = loss[0]
        train_losses += loss.item()
        model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5)
        optimizer.step()
        scheduler.step()

    train_loss = float(train_losses) / len(train_loader)
    print("Epoch: {}, train loss: {}".format(epoch, train_loss))


def eval_epoch(device, data_loader, model, epoch, mode="dev"):
    """
    used for evaluate the dev set or test set
    """
    model.eval()
    pred_tags = []
    gold_tags = []
    pred_labels = []
    true_labels = []

    for _, batch_samples in enumerate(tqdm(data_loader)):
        batch_data, _, batch_masks, batch_labels, sentence, origional_sentences, origional_labels, length, offset = batch_samples
        # shift tensors to GPU if available
        batch_data = batch_data.to(device)
        batch_masks = batch_masks.to(device)
        batch_labels = batch_labels.to(device)

        batch_output = model(batch_data,
                            token_type_ids=None, 
                            attention_mask=batch_masks)
        batch_output= batch_output[0].detach().cpu().numpy()
        batch_labels = batch_labels.to('cpu').numpy()
        pred_tags.extend([[id2label.get(idx.item()) for idx in indices] for indices in np.argmax(batch_output, axis=2)])
        gold_tags.extend([[id2label.get(idx.item()) if idx.item() >= 0 else "O" for idx in indices] for indices in batch_labels])

        for i in range(len(length)):
            data_tobe_convert = {}
            data_tobe_convert["length"] = length[i]
            data_tobe_convert["tags"] = ["O"] + pred_tags[i]
            data_tobe_convert["offset"] = offset[i]
            data_tobe_convert["sentence"] = origional_sentences[i]
            data_tobe_convert["words"] = sentence[i]
            pred_labels.append(get_char_level_label(data_tobe_convert))
            true_labels.append(origional_labels[i])

    scores = f1_score(length, pred_labels, true_labels)
    if mode == "dev":
        print("Epoch: {}, f1: {}".format(epoch, scores * 100))
    else:
        print("Test f1: {}".format(scores * 100))


def train_model(device, train_loader, dev_loader, model, optimizer, scheduler, epoch_num):
    """
    used for training bert model
    """
    for epoch in range(1, epoch_num + 1):
        train_epoch(device, train_loader, model, optimizer, scheduler, epoch)
        eval_epoch(device, dev_loader, model, epoch, mode="dev")

    print("Training Finished!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True, help="base path to load data")
    args = parser.parse_args()

    training_set = get_dataset("%s/tsd_train.csv" % args.base_path, split="train", flair_model_path=None)
    trial_set = get_dataset("%s/tsd_trial.csv" % args.base_path, split="trial", flair_model_path=None)
    test_set = get_dataset("%s/tsd_test.csv" % args.base_path, split="test", flair_model_path=None)

    train_loader = DataLoader(training_set, batch_size=32, shuffle=False, collate_fn=trial_set.collate_fn)
    trial_loader = DataLoader(trial_set, batch_size=32, shuffle=False, collate_fn=trial_set.collate_fn)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=trial_set.collate_fn)

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

    epoch_num = 50
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4, correct_bias=False)
    train_steps_per_epoch = len(training_set) // 32
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=train_steps_per_epoch,
                                                num_training_steps=epoch_num * train_steps_per_epoch)
    print("--------Start Training!--------")
    train_model(device, train_loader, trial_loader, model, optimizer, scheduler, epoch_num)

    print("--------Start Testing!--------")
    eval_epoch(device, test_loader, model, mode="test")