import os
import urllib.request
import zipfile
import tarfile
import re 
import torch.nn.functional as F
import torchtext 
from janome.tokenizer import Tokenizer 
from gensim.models import KeyedVectors

def tokenize(text):
    tokenizer = Tokenizer() 
    return [tok for tok in tokenizer.tokenize(text, wakati=True)]

def preprocessing_text(text):
    text = re.sub("\r", "", text)
    text = re.sub("\n", "", text)
    text = re.sub("  ", "", text)
    text = re.sub(" ", "", text)
    text = re.sub(r"[0-9 ０-９]", "0", text)
    return text 

def preprocess(text):
    text = preprocessing_text(text)
    out = tokenize(text)
    return out

def download_weights():
    '''
    word2vec, fasttext の事前学習済み重みのダウンロード
    '''
    print("downloading word2vec ...")
    data_dir = "./data/"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # word2vec
    url = "http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/data/20170201.tar.bz2"
    save_path = "./data/20170201.tar.bz2"
    if not os.path.exists(save_path):
        urllib.request.urlretrieve(url, save_path)
    tar = tarfile.open('./data/20170201.tar.bz2', 'r|bz2')
    tar.extractall('./data/')  
    tar.close()  

    print("done")
    print("converting...")

    # torchtext で扱える形式に変換
    model = KeyedVectors.load_word2vec_format(
        "./data/entity_vector/entity_vector.model.bin", binary=True
    )
    model.wv.save_word2vec_format("./data/japanese_word2vec_vectors.vec")

    print("done")

    # fasttext
    print("downloading fasttext")
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
    save_path = "./data/wiki-news-300d-1M.vec.zip"
    if not os.path.exists(save_path):
        urllib.request.urlretrieve(url, save_path)
    zip = zipfile.ZipFile("./data/wiki-news-300d-1M.vec.zip")
    zip.extractall("./data/")  
    zip.close()  


if __name__ == "__main__":

    download_weights()

    '''
    データセットの作成
    '''

    max_length = 25 

    TEXT = torchtext.data.Field(
        sequential=True, # 可変長かどうか
        tokenize=preprocess, # 前処理
        use_vocab=True, # 単語をボキャブラリーに追加するかどうか
        lower=True, # アルファベットを小文字にするか
        include_lengths=True,  # 文章の単語数データを保持するか
        batch_first=True,
        fix_length=max_length # padding して固定長にする
    )

    LABEL = torchtext.data.Field(
        sequential=False,
        use_vocab=False
    )

    train_ds, val_ds, test_ds = torchtext.data.TabularDataset.splits(
        path = "./data/",
        train = "text_train.tsv",
        validation = "text_val.tsv",
        test = "text_test.tsv",
        format = "tsv",
        fields = [("Text", TEXT), ("Label", LABEL)]
    )

    '''
    分散表現を用いてボキャブラリー作成
    '''
    vector = "word2vec"

    if vector == "word2vec":

        japanese_word2vec_vectors = torchtext.vocab.Vectors(
            name = "./data/japanese_word2vec_vectors.vec"
        )

        TEXT.build_vocab(
            train_ds, 
            vectors = japanese_word2vec_vectors,
            min_freq=1 # min_freq で出現回数に応じた足切り
            ) 

        print("vectors dim : {}".format(TEXT.vocab.vectors.shape))

        print(TEXT.vocab.stoi)

        # 姫 - 女性 + 男性
        tensor_calc = TEXT.vocab.vectors[41] - TEXT.vocab.vectors[38] + TEXT.vocab.vectors[46]

        print("女王: ", F.cosine_similarity(tensor_calc, TEXT.vocab.vectors[39]), dim=0)
        print("王: ", F.cosine_similarity(tensor_calc, TEXT.vocab.vectors[44]), dim=0)
        print("王子: ", F.cosine_similarity(tensor_calc, TEXT.vocab.vectors[45]), dim=0)
        print("機械学習: ", F.cosine_similarity(tensor_calc, TEXT.vocab.vectors[43]), dim=0)


    