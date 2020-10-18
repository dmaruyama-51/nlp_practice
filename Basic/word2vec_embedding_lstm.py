import os
import urllib.request
import tarfile
import numpy as np 
import pandas as pd
import MeCab
from glob import glob

from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import train_test_split

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import torchtext 

import pytorch_lightning as pl 
from pytorch_lightning import Trainer

def download_data():
    if not os.path.exists("./data/"):
      os.mkdir("./data/")
    url = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
    save_path = "./data/ldcc-20140209.tar.gz"
    if not os.path.exists(save_path):
        urllib.request.urlretrieve(url, save_path)
    tar = tarfile.open('./data/ldcc-20140209.tar.gz')
    tar.extractall('./data/')  
    tar.close() 

def prepare_dataset():
  dirs = glob('data/text/**/')
  texts, labels = [], []
  label_dicts = {}
  for (label, dir) in enumerate(dirs):
      label_dicts[label] =dir.split("/")[-2]
      filepaths = glob('{}/*'.format(dir))
      for filepath in filepaths:
          with open(filepath, encoding='utf-8') as f:
            text = f.readlines()[2:]
            text = ''.join(text)
            texts.append(text)
          labels.append(label)

  df = pd.DataFrame({"TEXT": texts, "LABEL": labels})
  df["LABEL_NAME"] = [label_dicts[label] for label in df["LABEL"]]

  train_val_df, test_df = train_test_split(df, test_size=0.2, stratify=df["LABEL"], random_state=0)
  train_df, val_df = train_test_split(train_val_df, test_size=0.3, stratify=train_val_df["LABEL"], random_state=0)

  train_df.to_csv("data/livedoor_train.csv", index=False)
  val_df.to_csv("data/livedoor_val.csv", index=False)
  test_df.to_csv("data/livedoor_test.csv", index=False)

def tokenize(text):
  text = text.replace('\u3000', '')
  text = text.replace('\n', '')
  mecab = MeCab.Tagger("-Owakati")
  res = mecab.parse(text).strip().split()
  return res

class TrainNet(pl.LightningModule):

    def train_dataloader(self):
        return torchtext.data.BucketIterator(train_ds, batch_size=self.batch_size, shuffle=True)

    def training_step(self, batch, batch_idx):
        x, t = batch.TEXT, batch.LABEL 
        y = self.forward(x)
        loss = self.lossfun(y, t)
        results = {'loss': loss}
        self.log("loss", loss, on_epoch=True)
        return results

class ValidationNet(pl.LightningModule):

    def val_dataloader(self):
        return torchtext.data.BucketIterator(val_ds, batch_size=self.batch_size, shuffle=False)

    def validation_step(self, batch, batch_idx):
        x, t = batch.TEXT, batch.LABEL
        y = self.forward(x)
        loss = self.lossfun(y, t)
        y_label = torch.argmax(y, dim=1)
        acc = torch.sum(t == y_label) * 1.0 / len(t)
        results = {'val_loss': loss, 'val_acc': acc}
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)
        return results

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        results = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return results

class TestNet(pl.LightningModule):

    def test_dataloader(self):
        return torchtext.data.BucketIterator(test_ds, batch_size=self.batch_size, shuffle=False)

    def test_step(self, batch, batch_idx):
        x, t = batch.TEXT, batch.LABEL
        y = self.forward(x)
        loss = self.lossfun(y, t)
        y_label = torch.argmax(y, dim=1)
        acc = torch.sum(t == y_label) * 1.0 / len(t)
        results = {'test_loss': loss, 'test_acc': acc}
        return results

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        results = {'test_loss': avg_loss, 'test_acc': avg_acc}
        return results

class LSTMClassifier(TrainNet, ValidationNet, TestNet):

    def __init__(self, vocab_size=8684, embed_dim=200, hidden_dim=100, batch_size=256, num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 9)

        self.embed.weight = nn.Parameter(TEXT.vocab.vectors)
        self.embed.weight.requires_grad = False

    def lossfun(self, y, t):
        return F.cross_entropy(y, t)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-4)

    def forward(self, x):
        # (batch_size, seq_length)
        x = self.embed(x) 
        # (batch_size, seq_length, embed_dim)
        hs, (h, c) = self.lstm(x)
        out = self.fc(h.view(hs.size(0), -1))
        return out

if __name__ == "__main__":

    '''
    学習データと教師データを準備
    '''

    download_data()
    prepare_dataset()

    '''
    word2vec で分散表現獲得
    '''
    train_df = pd.read_csv("data/livedoor_train.csv")
    word_collect = []
    for text in train_df["TEXT"]:
        word_collect.append(tokenize(text))
    
    model = Word2Vec(word_collect, size=200, window=10, min_count=20)
    model.wv.save_word2vec_format("word2vec_vectors.vec")

    '''
    torchtext で 事前学習済み埋め込みベクトルを使用
    '''
    TEXT = torchtext.data.Field(
        tokenize = tokenize,
        use_vocab=True,
        include_lengths=False,
        lower=True,
        batch_first=True,
        fix_length=200
    )

    LABEL = torchtext.data.Field(
        sequential=False,
        use_vocab=False
    )

    train_ds, val_ds, test_ds = torchtext.data.TabularDataset.splits(
        path = "data",
        train = "livedoor_train.csv",
        validation = "livedoor_val.csv",
        test = "livedoor_test.csv",
        format = "csv",
        fields = [("TEXT", TEXT), ("LABEL", LABEL)],
        skip_header = True
    )
    
    word2vec_vectors = torchtext.vocab.Vectors(name="word2vec_vectors.vec")
    TEXT.build_vocab(train_ds, vectors=word2vec_vectors, min_freq=20)


    '''
    pytorch-lightning で学習
    '''
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    model = LSTMClassifier(
        vocab_size=len(TEXT.vocab),
        hidden_dim=100,
        batch_size=256
        )
    trainer = Trainer(gpus=1, max_epochs=300)
    trainer.fit(model)


