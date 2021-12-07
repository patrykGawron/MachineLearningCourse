import torch
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import AG_NEWS
import mlflow.sklearn
import pandas as pd
import random
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from torch import nn


class KlasyfikatorTekstu3000(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(KlasyfikatorTekstu3000, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, 4)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=5)

    def forward(self, text, offsets):
        out = self.embedding(text, offsets)
        return self.fc(out)


class Laboratory:
    def __init__(self):
        self.todo1()
        self.todo2()

    def todo1(self):
        self.tokenizer = get_tokenizer("basic_english")
        tekst = "I'm having a wonderful time at WZUM laboratories!"
        # vocab = {word: i for i, word in enumerate(set(tokenizer(tekst)))}
        # print(vocab)

        train_data = AG_NEWS(split='train')
        self.vocab = build_vocab_from_iterator(self.yield_tokens(train_data), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])
        print((self.vocab(self.tokenizer(tekst))))

        self.text_pipeline = lambda x: self.vocab(self.tokenizer(x))

    def todo2(self):
        from torch.utils.data import DataLoader
        dataset = AG_NEWS(split='train')
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=self.collate_batch)

        self.model = KlasyfikatorTekstu3000(len(self.vocab), 5)
        for labels, texts, offsets in dataloader:
            print(self.model(texts, offsets))
            exit()

    def todo6(self, dataloader):
        self.model.train()

        for label, text, offset in dataloader:
            self.model.optimizer.zero_grad()
            pred_label = self.model(text, offset)
            loss = self.model.loss_function(pred_label, label)
            loss.backward()


    def yield_tokens(self, iter):
        for _, text in iter:
            yield self.tokenizer(text)

    def collate_batch(self, batch):
        label_list, text_list, offset = [], [], [0]

        for label, text in batch:
            label_list.append(label-1)
            processed_text = torch.tensor(self.text_pipeline(text), dtype=torch.int64)
            text_list.append(processed_text)
            offset.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list = torch.cat(text_list)
        offset = torch.tensor(offset[:-1]).cumsum(dim=0)

        return label_list, text_list, offset

if __name__ == "__main__":
    lab = Laboratory()