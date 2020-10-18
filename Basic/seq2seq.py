import numpy as np
import pandas as pd
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchtext 
import pytorch_lightning as pl 
from pytorch_lightning import Trainer

from tqdm import tqdm

np.random.seed(0)
device = "cuda" if torch.cuda.is_available else "cpu"

def generate_dataset(filename, n_sample):
    srcs, trgs = [], []
    n_sample = int(n_sample / 5)

    symb = '+'

    # ４桁
    values = np.random.randint(1, 10000, (n_sample*2, 2))
    for (val1, val2) in values:
        src = str(val1) + symb + str(val2)
        trg = '{:.0f}'.format(eval(src))
        srcs.append(src)
        trgs.append(trg)

    # ３桁
    values = np.random.randint(1, 1000, (n_sample, 2))
    for (val1, val2) in values:
        src = str(val1) + symb + str(val2)
        trg = '{:.0f}'.format(eval(src))
        srcs.append(src)
        trgs.append(trg)

    # ２桁
    values = np.random.randint(1, 100, (n_sample, 2))
    for (val1, val2) in values:
        src = str(val1) + symb + str(val2)
        trg = '{:.0f}'.format(eval(src))
        srcs.append(src)
        trgs.append(trg)

    # １桁
    values = np.random.randint(1, 10, (n_sample, 2))
    for (val1, val2) in values:
        src = str(val1) + symb + str(val2)
        trg = '{:.0f}'.format(eval(src))
        srcs.append(src)
        trgs.append(trg)

    df = pd.DataFrame({'src': srcs, 'trg': trgs})
    df.to_csv(filename, header=None, index=None)
    return df

class TrainNet(pl.LightningModule):

    def train_dataloader(self):
        return torchtext.data.BucketIterator(train, self.batch_size, shuffle=True)

    def training_step(self, batch, batch_idx):
        x= batch.x
        t = batch.t
        y, t = self.forward(x, t)
        loss = self.lossfun(y, t)
        self.log("train_loss", loss)
        return loss

class ValidationNet(pl.LightningModule):

    def val_dataloader(self):
        return torchtext.data.BucketIterator(val, self.batch_size)

    def validation_step(self, batch, batch_idx):
        x= batch.x
        t = batch.t
        y, t = self.forward(x, t)
        loss = self.lossfun(y, t)
        self.log("val_loss", loss)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log("epoch_val_loss", avg_loss)
        return {"epoch_val_loss": avg_loss}

class Encoder(pl.LightningModule):

    def __init__(self, n_input, n_embed, n_hidden, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.embed = nn.Embedding(n_input, n_embed, padding_idx=1)
        self.lstm = nn.LSTM(n_embed, n_hidden, n_layers, bidirectional=True)

    def forward(self, x):
        x = self.embed(x)
        x, (h, c) = self.lstm(x)
        return h, c

class Decoder(pl.LightningModule):

    def __init__(self, n_output, n_embed, n_hidden, n_layers):
        super().__init__()
        self.output_dim = n_output
        self.embed = nn.Embedding(n_output, n_embed, padding_idx=1)
        self.lstm = nn.LSTM(n_embed, n_hidden, n_layers, bidirectional=True)
        self.fc = nn.Linear(n_hidden, n_output)

    def forward(self, x, h, c):
        x = x.unsqueeze(0)
        x = self.embed(x)
        x, (h, c) = self.lstm(x, (h, c))
        y = self.fc(h[-1])
        return y, h, c

class Seq2Seq(TrainNet, ValidationNet):

    def __init__(self, *args, batch_size=256, num_workers=8):
        super().__init__()
        self.encoder = Encoder(n_input, n_embed_enc, n_hidden, n_layers)
        self.decoder = Decoder(n_output, n_embed_dec, n_hidden, n_layers)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def lossfun(self, y, t):
        return F.cross_entropy(y, t, ignore_index=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x, t):
        max_len, batch_size = t.shape
        t_vocab_size = self.decoder.output_dim
        y = torch.zeros(max_len - 1, batch_size, t_vocab_size, device=device)
        h, c = self.encoder(x)
        inputs = t[0, :]

        for i in range(max_len - 1):
            _y, h, c = self.decoder(inputs, h, c)
            y[i] = _y
            top1 = _y.max(1)[1]
            inputs = top1

        y = y.view(-1, y.shape[-1])
        t = t[1:].view(-1)

        return y, t

def translate(x, vocab):
    text = ''
    for s in x:
        text += vocab.itos[s]
    text = text.replace('<sos>', '')
    text = text.replace('<eos>', '')
    text = text.replace('<pad>', '')
    return text

def predict(x, vocab):
  answer = []
  net.eval().to(device)
  h, c = net.encoder(x)
  id_sos = vocab.stoi["<sos>"]
  x = torch.tensor([id_sos]).to("cuda")
  while True:
    x, h, c = net.decoder(x, h, c)
    x = x.max(1)[1]
    if vocab.itos[x] != "<eos>":
      answer.append(x)
    else:
      break
  return translate(answer, vocab)


if __name__ == "__main__":

    # データセット作成
    df_train = generate_dataset('train.csv', 100000)
    df_val = generate_dataset('val.csv', 10000)
    df_test = generate_dataset('test.csv', 10000)

    def tokenize(text):
        return [tok for tok in text]

    field_x = torchtext.data.Field(
        tokenize=tokenize,
    )

    field_t = torchtext.data.Field(
        tokenize=tokenize,
        init_token="<sos>",
        eos_token="<eos>"
    )

    train, val, test = torchtext.data.TabularDataset.splits(
        path = ".",
        train = "train.csv",
        validation = "val.csv",
        test = "test.csv",
        format = "csv",
        fields = [("x", field_x), ("t", field_t)]
    )

    field_x.build_vocab(train)
    field_t.build_vocab(train)

    # ハイパーパラメータ
    n_input = len(field_x.vocab)
    n_output = len(field_t.vocab)
    n_embed_enc = 200
    n_embed_dec = 200
    n_hidden = 200
    n_layers = 2
    gpus = 1
    max_epochs = 20

    # 学習
    print("training ...")
    torch.manual_seed(0)
    net = Seq2Seq(n_input, n_embed_enc, n_embed_dec, n_hidden, n_layers)
    trainer = Trainer(gpus=gpus, max_epochs=max_epochs)
    trainer.fit(net)

    print("done")


    '''
    正解率の算出
    '''
    test_dataloader = torchtext.data.BucketIterator(test, batch_size=256)
    results = {
        "question": [], 
        "answer": [],
        "predict": []
        }

    print("test start...")
    for batch in tqdm(test_dataloader):
        for i in range(batch.x.shape[1]):
            x, t = batch.x[:, i:i+1].to(device), batch.t[:, i:i+1].to(device)
            question = translate(x, field_x.vocab)
            answer = translate(t, field_t.vocab)
            pred = predict(x, field_t.vocab)

            results["question"].append(question)
            results["answer"].append(answer)
            results["predict"].append(pred)
        
    df = pd.DataFrame(results)

    df['hantei'] = df['answer'] == df['predict']
    accuracy = df['hantei'].sum() / len(df)
    print("")
    print("accuracy : {} %".format(accuracy*100))