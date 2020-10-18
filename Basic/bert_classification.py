import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import os
import urllib.request
import tarfile
from glob import glob

import linecache
import warnings
warnings.filterwarnings('ignore')

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchtext 
from transformers.modeling_bert import BertModel 
from transformers.tokenization_bert_japanese import BertJapaneseTokenizer

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
    categories = [name for name in os.listdir("data/text")]

    datasets = pd.DataFrame(columns=["title", "category"])
    for cat in categories:
        path = "data/text/" + cat + "/*.txt"
        files = glob(path)
        for text_name in files:
            title = linecache.getline(text_name, 3)
            s = pd.Series([title, cat], index=datasets.columns)
            datasets = datasets.append(s, ignore_index=True)

    # データをシャッフル
    livedoor_data = datasets.sample(frac=1).reset_index(drop=True)

    categories = list(set(livedoor_data['category']))
    id2cat = dict(zip(list(range(len(categories))), categories))
    cat2id = dict(zip(categories, list(range(len(categories)))))

    livedoor_data['category_id'] = livedoor_data['category'].map(cat2id)
    livedoor_data = livedoor_data.sample(frac=1).reset_index(drop=True)

    livedoor_data = livedoor_data[['title', 'category_id']]

    train_df, test_df = train_test_split(livedoor_data, train_size=0.8)
    train_df.to_csv('data/livedoor_train.csv', index=False, header=None)
    test_df.to_csv('data/livedoor_test.csv', index=False, header=None)

class TrainNet(pl.LightningModule):

    def train_dataloader(self):
        return torchtext.data.Iterator.splits(train_ds, batch_sizes=self.batch_size, repeat=False, sort=False)

    def training_step(self, batch, batch_idx):
        x= batch.x
        t = batch.t
        y, t = self.forward(x, t)
        loss = self.lossfun(y, t)
        self.log("train_loss", loss, on_epoch=True)
        return loss

class ValidationNet(pl.LightningModule):

    def val_dataloader(self):
        return torchtext.data.Iterator.splits(val_ds, batch_sizes=self.batch_size, repeat=False, sort=False)

    def validation_step(self, batch, batch_idx):
        x= batch.x
        t = batch.t
        y, t = self.forward(x, t)
        loss = self.lossfun(y, t)
        y_label = torch.argmax(y, dim=1)
        acc = torch.sum(t == y_label) * 1.0 / len(t)
        results = {'test_loss': loss, 'test_acc': acc}
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)
        return results

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        results = {'test_loss': avg_loss, 'test_acc': avg_acc}
        return results


class BertClassifier(TrainNet, ValidationNet):

    def __init__(self, batch_size=256, num_workers=8):
        super().__init__()
        self.model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        self.linear = nn.Linear(768, 9)
        self.batch_size = batch_size
        self.num_workers = num_workers

        for param in self.model.parameters():
            param.requires_grad = False
        
        for param in self.model.encoder.layer[-1].parameters():
            param.requires_grad = True

        for param in self.model.linear.parameters():
            param.requires_grad = True

    def lossfun(self, y, t):
        return F.cross_entropy(y, t)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def forward(self, x, t):
        vec, _ = self.bert(input_ids, output_attention=False)
        vec = vec[:, 0, :].view(-1, 768)
        out = self.linear(vec)
        return out

def bert_tokenizer(text):
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    return tokenizer.encode(text, return_tensors='pt')[0]


if __name__ == "__main__":

    print("preparing datasets...")
    download_data()

    prepare_dataset()
    print("done!")

    print("building BertModel...")
    field_x = torchtext.data.Field(
        tokenize=bert_tokenizer,
        use_vocab=False,
        include_lengths=True,
        batch_first=True,
        pad_token=0
        )

    field_t = torchtext.data.Field(
        sequential=False,
        use_vocab=False
        )
    
    train_ds, val_ds = torchtext.data.TabularDataset.splits(
        path='data',
        train='livedoor_train.csv',
        test='livedoor_test.csv',
        format='csv',
        fields=[('x', field_x), ('t', field_t)]
        )
    
    model = BertClassifier(batch_size=32)
    print("done")
    print("training...")
    trainer = Trainer(gpus=1, max_epochs=1)
    trainer.fit()
    print("done!")