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
from torch.utils.data import DataLoader


class KlasyfikatorTekstu3000(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(KlasyfikatorTekstu3000, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, 4)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, 0.1)

    def forward(self, text, offsets):
        out = self.embedding(text, offsets)
        return self.fc(out)


class Laboratory:
    def __init__(self):
        self.ag_news_label = {
            1: "World",
            2: "Sports",
            3: "Business",
            4: "Sci/Tec"
        }
        self.todo1()
        self.todo2()
        self.test()
        news = "Messi vs Ronaldo fans, UEFA from the chaotic Champions League last-16 draw"
        print(self.ag_news_label[self.predict(news)])

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
        dataset = AG_NEWS(split='train')
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=self.collate_batch)

        self.model = KlasyfikatorTekstu3000(len(self.vocab), 5)

    #TODO: 6
    def todo6(self, dataloader):
        self.model.train()

        total_acc, total_count = 0, 0
        for idx, (label, text, offset) in enumerate(dataloader):
            self.model.optimizer.zero_grad()
            pred_label = self.model(text, offset)
            loss = self.model.loss_function(pred_label, label)
            loss.backward()

            self.model.optimizer.step()

            total_acc += (pred_label.argmax(1) == label).sum().item()
            total_count += label.size(0)

            if idx % 500 == 0:
                print("Count: ", idx, "Acc: ", total_acc / total_count * 100, " %")
                total_acc, total_count = 0, 0

    #TODO: 6
    def evaluate(self, dataloader):
        self.model.eval()

        with torch.no_grad():
            total_acc, total_count = 0, 0
            for idx, (label, text, offset) in enumerate(dataloader):
                pred_label = self.model(text, offset)

                total_acc += (pred_label.argmax(1) == label).sum().item()
                total_count += label.size(0)

        return total_acc / total_count * 100

    def test(self):
        train_iter, test_iter = AG_NEWS()
        from torchtext.data.functional import to_map_style_dataset
        train_dataset = to_map_style_dataset(train_iter)
        test_dataset = to_map_style_dataset(test_iter)

        EPOCHS = 10
        BATCH_SIZE = 64

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=self.collate_batch)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=self.collate_batch)

        for epoch in range(1, EPOCHS + 1):
            self.todo6(train_dataloader)
            acc_val = self.evaluate(test_dataloader)
            print("="*20)
            print("\n", "Acc epoch ", epoch, ": ", acc_val, "\n")
            print("="*20)

    def predict(self, text):
        self.model = self.model.to('cpu')
        with torch.no_grad():
            text = torch.tensor(self.text_pipeline(text))
            output = self.model(text, torch.tensor([0]))
            return output.argmax(1).item() + 1



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
