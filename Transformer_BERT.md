

[TOC]

## トレンド

- 事前学習言語モデルのファインチューニング（BERT, RoBERTa など）
  - GPT-3 はファインチューニングすら不要
- 多段階事前学習
  - 事前学習：大規模なラベルなしデータで自己教師あり学習に基づき事前学習
  - 中間タスク（Intermidiate Task Training）
  - ファインチューニング：タスクのラベルありで教師あり学習
- BERT → RoBERTa がデフォルトになってきた？



## Seq2seq, Attention

自己回帰言語モデル。1単語順番に読み込む。（Self Attention では一発で単語同士の依存関係を学習）

seq2seq では、エンコーダーが受け取る入力時系列のデータは、最終的に 1 つの固定長ベクトルで表現される

↓

・本来は各時刻によって過去のどの時刻を重視すべきは異なるはず

・どんなに長い系列でも、１つの固定長で表現＝適切に表現できていない



## ELMo

- Embeddings from Language Models 

- 文脈を考慮した分散表現を獲得する手法



## Transformer 

以下のコンポーネントで構成

- Embedding

- Positional Encoding

  - Attention で系列間の関連性を学習することはできても、系列データの順序、単語の並び順は考慮されていない。（LSTM ベースの Attention では問題ない。）
  - Transformer では LSTM のような再帰的な処理の代わりに、Positional Encoding で系列データに直接順序・位置関係の情報を埋め込むことを実現
  - 事前に以下の式で作成した行列PEを単純に足す処理。
  - ![スクリーンショット 2020-10-06 20.51.12](/Users/kikagaku/Library/Application Support/typora-user-images/スクリーンショット 2020-10-06 20.51.12.png)
  - <img src="/Users/kikagaku/Library/Application Support/typora-user-images/スクリーンショット 2020-10-06 20.51.38.png" alt="スクリーンショット 2020-10-06 20.51.38" style="zoom:80%;" />

- (Masked) Multi Head Attention

  ![スクリーンショット 2020-10-06 20.46.45](/Users/kikagaku/Library/Application Support/typora-user-images/スクリーンショット 2020-10-06 20.46.45.png)

  - Scaled Dot-Product Attention を複数並列に行う。複数の時点から情報を抽出することが期待される。

- Add & Norm

  - Add : 残差接続（residual connection） 

    Add(x, SubLayer(x)) = x + SubLayer(x)

    ![スクリーンショット 2020-10-06 20.44.13](/Users/kikagaku/Library/Application Support/typora-user-images/スクリーンショット 2020-10-06 20.44.13.png)

  - Norm : Layer Normalization 

    ミニバッチ単位で正規化ではなく、特徴量の次元（ニューロン数）における平均分散を用いた正規化

- Feed Forward

- Linear 

- Softmax







## BERT 



- アーキテクチャ
  - BERT BASE : Encoder × 12 
  - BERT LARGE : Encoder × 24
- input 
  - 分かち書きして、トークン化した数字（トークンID）を入力にする。
  - MAX 512 
- 2 つのタスクで事前学習
  - Masked Language Model : CBOW の拡張。ランダムにいくつかの単語をマスク化し、それを残りの単語すべてを使って推定するタスク
    - 全体の 80% は [Mask] トークン
    - 10 % はランダムに選んだトークンで置き換える
    - 10 % は変化なし
  - Next Sentence Predicition : 512単語で [SEP] を挟んで2つの文章が入力となる。「連続的に存在する意味があって関係が深い文章」の場合と、「まったく関係がなく文脈のつながりがない2つの文章」の2パターンを用意。どちらであるか推定するタスク
    - →文同士の関係性を考慮
- Input: 
  - 先頭に [CLS] トークン
  - 文の境目には [SEP] トークン



## How to Use Huggingface 



### 基本

```python
# モデルの構築
model = AutoModel.from_pretrained(MODEL_NAME)

# tokenizer の準備
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 分かち書き→IDに変換→Special Token付与 [CLS] [SEP] [PAD]など
TEXT = "This is an input example")
token_ids = tokenizer(TEXT, return_tensors="pt", padding=True)

# 複数の場合 → padding
tokens_ids = tokenizer(
    ["This is a sample", "This is another longer sample text"], 
    return_tensors="pt",
    padding=True  
)

# ID→文字に戻す
print(tokenizer.convert_ids_to_tokens(tokens["input_ids"][0]))

'''
['[CLS]', 'This', 'is', 'a', 'sample', '[SEP]', '[PAD]', '[PAD]']
'''

# アウトプットの形状
output, pooled = model(**token_ids)
# output : (batch_size, num_token_ids, representation_dims)
# pooled : (batch_size, representation_dims)
```



### pipeline



以下のタスクの高レベル API

- Sentence Classification (Sentiment Analysis)：文書・文章単位の分類
- Token Classification (NER, Part-of-Speech Tagging) : トークン単位の分類
- Question-Answering : IN ( qustion, context) -> OUT (answer)
- Mask-Filling : 一部がマスクされた文章を入力して、入る可能性のある単語を提案
- Summarization : 要約
- Translation : 翻訳
- Feature Extraction : 分散表現の獲得



```python
from transformers import pipeline

# Using default model and tokenizer for the task
pipeline("<task-name>")

# Using a user-specified model
pipeline("<task-name>", model="<model_name>")

# Using custom model/tokenizer as str
pipeline('<task-name>', model='<model name>', tokenizer='<tokenizer_name>')
```



<task-name>

'sentiment-analysis', 'ner', 'question-answering', 'fill-mask', 'summarization', 'translation_en_to_fr', text-generation, 



※ translation では、現在 'translation_en_to_fr', 'translation_en_to_de', 'translation_en_to_ro' のみサポート（モデルは T5）



## Text Generation 

GPT-2 で文章生成する際の戦略

- greedy search : 最も確率の高い単語を採用する
- ビームサーチ：複数単語先まで考慮して最も高い確率のシーケンスを採用する。
- top-K sampling : 



```python
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

# 入力テキスト
input_ids = tokenizer.encode('I enjoy walking with my cute dog', return_tensors='tf')

# greedy search 
greedy_output = model.generate(input_ids, max_length=50)

# beam search : EOS token 出てくるまで 5 つ探索
beam_output = model.generate(
    input_ids,  
    max_length=50, 
    num_beams=5, 
    early_stopping=True
)

# beam search : 繰り返し表現, 2gram の制約をつける
beam_output = model.generate(
    input_ids, 
    max_length=50, 
    num_beams=5, 
    no_repeat_ngram_size=2, 
    early_stopping=True
)


# beam search : set return_num_sequences > 1
beam_outputs = model.generate(
    input_ids, 
    max_length=50, 
    num_beams=5, 
    no_repeat_ngram_size=2, 
    num_return_sequences=5, 
    early_stopping=True
)

# top-k 
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=50, 
    top_k=50
)

```



## GPT-3



- BERTなど：事前学習→ファインチューニングしてパラメータをタスクに最適化する

- GPT-3 : 事前学習後、パラメータは変わらない

  - 単方向アーキテクチャ
  - in-context learning、文脈内学習
  - 事前学習言語モデルに、テキスト形式でタスク定義と例をいくつか与えるのみでOK
    - Few shot : 10 ~ 100 くらいの example
    - One-shot : 1 つの example 
    - zero-shot : task description のみ

  ```
  Translate English to French: # task description
  
  sea otter => loutre de mer # example
  
  cheese => # prompt
  ```

- GPT-2 からの進化
  - モデルが超巨大化した
  - パラメータ数は約 1750 億個
- 得意なタスク・苦手なタスク
  - ◯ LAMBADA : 文章が与えられた時に、最後の単語を予測する。
    - Alice was friends with Bob. Alice went to visit her friend xxx => Bob
  - ◯ 文章生成
  - ✕ 機械読解：質問に対する回答（多肢選択型、抽出型など）
  - ✕ 一般常識推論：「チーズを冷蔵庫の中に入れたら、溶けるでしょうか？」
  - ✕ 自然言語推論：「何人かの男がサッカーをしている」→「何人かの男がスポーツをしている」
  - 2つの文の比較や、長い文章を呼んだ後に質問に回答するようなタスクはあまり得意ではない。





















