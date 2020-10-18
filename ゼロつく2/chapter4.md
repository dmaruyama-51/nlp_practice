## word2vecの高速化

### word2vecの改良①
SimpleCBOWの問題点
- 入力データ(one-hot表現)と重み行列W_inのdot積の計算: (1, 1000000)*(1000000, 100)
    - 語彙数が100万あれば、100万の要素を占めるメモリサイズが必要
    

#### Embeddingレイヤー


```python
import numpy as np
W = np.arange(21).reshape(7,3)
```


```python
W
```




    array([[ 0,  1,  2],
           [ 3,  4,  5],
           [ 6,  7,  8],
           [ 9, 10, 11],
           [12, 13, 14],
           [15, 16, 17],
           [18, 19, 20]])




```python
W[2]
```




    array([6, 7, 8])




```python
idx = np.array([1, 0, 3, 0])
W[idx]
```




    array([[ 3,  4,  5],
           [ 0,  1,  2],
           [ 9, 10, 11],
           [ 0,  1,  2]])




```python
class Embedding:
    def __init__(self, W):
        self.param = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None
    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out
```


```python
def backward(self, dout):
    dW, = self.grads
    dW[...] = 0
    
    np.add.at(dW, self.idx, dout)
    return None
```

### word2vecの改良②
SimpleCBOWの問題点その２
- 中間層のニューロンと重み行列W_outの積
- Softmaxレイヤーの計算

#### 多値分類から二値分類へ
- Softmaxで多値分類にするのではなく、「ターゲットですか？Yes/No」の形にニューラルネットワークを変更する

テクニック
- dot積をとるのではなく、numpy配列でidxを指定して抜き出す


```python
class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None
    
    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)
        
        self.cache = (h, target_W)
        return out 
    
    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)
        
        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh
```

#### Negative Sampling
- 上記クラスでは、正例についだけしか学習できていない。負例についてはどうような結果になるか定かではない

⇓

- 正例をターゲットとした場合の損失を求める
- 同時に、負例をいくつかサンプリングし、その負例に対しても同様に損失を求める。
- それぞれのデータにおける損失を足し合わせ、その結果を最終的な損失とする

※すべての負例を扱わないのは計算コストの観点から

負例をどのようにサンプリングするか
- コーパスの確率分布に基づいてサンプリング（＝コーパス中でよく使われる単語は抽出されやすくし、あまり使われない単語は抽出されにくくする）



```python
#確率分布に基づくサンプリングの例
import numpy as np

words = ["you", "say", "goodbye", "I", "hello", "."]
p = [0.5, 0.1, 0.05, 0.2, 0.05, 0.1]
np.random.choice(words, p=p, size=5)
```




    array(['say', 'you', 'I', 'you', 'you'], dtype='<U7')



- 50%の確率でyouが出る、を考慮した出力になっている

word2vecのNegative Samplingでは、元の確率分布の各要素を0.75乗する
- 出現確率の低い単語に対して少し出る確率を高めてあげる効果


```python
p = [0.7, 0.29, 0.01]
new_p = np.power(p, 0.75)
new_p /= np.sum(new_p)
print(new_p)
```

    [0.64196878 0.33150408 0.02652714]
    

- 0.01 -> 0.026

### 改良版word2vecの学習


```python
from common.layers import Embedding
from ch04.negative_sampling_layer import NegativeSamplingLoss

class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size
        
        #重みの初期化
        W_in = 0.01 * np.random.randn(V, H).astype("f")
        W_out = 0.01 * np.random.randn(V, H).astype("f")
        
        #レイヤの作成
        self.in_layer = []
        for i in range(2 * window_size):
            layer = Embedding(W_in) 
            self.in_layer.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)
        
        #すべての重みと勾配を配列にまとめる
        layers = self.in_layer + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        
        #メンバ変数に単語の分散表現を設定
        self.word_vec = W_in
    
    def forward(self, contexts, target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h += 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return loss
    
    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout += 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None
            
```

- 比較用にSimpleCBOW


```python
class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size
        
        #重みの初期化
        W_in = 0.01 * np.random.randn(V, H).astype("f")
        W_out = 0.01 * np.random.randn(H, V).astype("f")
        
        #レイヤの作成
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()
        
        #すべての重みと勾配をリストにまとめる
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        
        #メンバ変数に単語の分散表現を設定
        self.word_vecs = W_in
    
    def forward(self, contexts, target):
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss
    
    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None
```

#### CBOWモデルの学習

PBTデータセットでは時間がかかるため省略

#### CBOWモデルの評価


```python
from common.util import most_similar
import pickle
```


```python
pkl_file = "ch04/cbow_params.pkl"
```


```python
with open(pkl_file, "rb") as f :
    params = pickle.load(f)
    word_vecs = params["word_vecs"]
    word_to_id = params["word_to_id"]
    id_to_word = params["id_to_word"]
querys = ["you", "year", "car", "toyota"]

for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
```

    
    [query] you
     we: 0.6103515625
     someone: 0.59130859375
     i: 0.55419921875
     something: 0.48974609375
     anyone: 0.47314453125
    
    [query] year
     month: 0.71875
     week: 0.65234375
     spring: 0.62744140625
     summer: 0.6259765625
     decade: 0.603515625
    
    [query] car
     luxury: 0.497314453125
     arabia: 0.47802734375
     auto: 0.47119140625
     disk-drive: 0.450927734375
     travel: 0.4091796875
    
    [query] toyota
     ford: 0.55078125
     instrumentation: 0.509765625
     mazda: 0.49365234375
     bethlehem: 0.47509765625
     nissan: 0.474853515625
    

word2vecは単に類似単語を取得するだけでなく、もっと複雑な関係性もとらえている
- 類推問題： king - man + woman = queen


```python
from common.util import analogy
```


```python
analogy("king", "man", "queen", word_to_id, id_to_word, word_vecs)
```

    
    [analogy] king:man = queen:?
     woman: 5.16015625
     veto: 4.9296875
     ounce: 4.69140625
     earthquake: 4.6328125
     successor: 4.609375
    


```python
analogy("take", "took", "go", word_to_id, id_to_word, word_vecs)
```

    
    [analogy] take:took = go:?
     went: 4.55078125
     points: 4.25
     began: 4.09375
     comes: 3.98046875
     oct.: 3.90625
    

### word2vecに関する残りのテーマ
#### word2vecを使ったアプリケーションの例

単語の分散表現化は転移学習と組み合わせることで効果絶大
- 自然言語処理タスク（テキスト分類、文書クラスタリング等）を行う際、単語の分散表現をゼロから獲得するようなことはほとんどしない
- 先に大きなコーパスで学習を行い、その学習済みの分散表現をタスクで使う

文章もベクトル化できる
- 単純なアイデアとしては、文章の各単語の分散表現を足し合わせる(bag-of-words,単語の順序を考慮しないモデル)
- RNNがここで生きてくるらしい

例）アプリ事業者の、お客様の声の分析
- 送られる声から、3種類の感情(pos, neutral, neg)に分類するシステムを作る
- word2vecで分散表現化→SVMやNNで分類タスク
- 不満を持つユーザーの声から順番に目を通すことができる

#### 単語ベクトルの評価方法

word2vecの評価は、それがシステムに組み込まれた最終的なアウトプット（分類の精度等）では図らない。それは分類器の評価。

⇓

単語の類似性や、類推問題による評価
- catとanimalの類似度は8, catとcarの類似度は2と事前に決めておき、それとword2vecによるコサイン類似度スコアを比較する
- king:queen = man:? のような類推問題の正解率

わかっていること
- モデルによって精度が異なる（コーパスに応じて最適なモデルを選ぶ）
- コーパスが大きいほど良い結果になる（ビッグデータは常に望まれる）
- 単語ベクトルの次元数は適度な大きさが必要（大きすぎても精度が悪くなる）：300くらい


