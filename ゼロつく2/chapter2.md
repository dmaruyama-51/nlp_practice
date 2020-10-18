### 自然言語処理とは
- 自然言語：人が話す言語
- 自然でな言語：コンピュータの世界の言語（プログラミング言語、マークアップ言語）

#### 活用事例
- 検索エンジン
- 機械翻訳
- 質問の応答システム
- IME（かな漢字変換）
- 文章の自動要約
- 感情分析

#### 単語の意味
コンピュータにどうやって単語の意味を理解させるか
- シソーラスによる手法(→nltk)
- カウントベースの手法
- 推論ベースの手法(→word2vec)

### シソーラス
人が定義した単語の意味を覚えさせる
- シソーラス＝類語辞書
- それぞれの単語の関係はグラフで表示

#### Wordnet
- NLP界隈で最も有名なシソーラス。1985年に開発スタートと歴史ある。
- 類義語取得、単語ネットワーク取得、単語間の類似度計算ができる

#### シソーラスの問題点
①時代の変化に対応するのが困難
- 新しい単語が生まれたら随時更新していかないといけない

②人の作業コストが高い
- 単語の関連付けまでしないといけないのでめっちゃ大変

③単語の細かなニュアンスを表現できない
- 「ヴィンテージ」「レトロ」は同じような意味合いを示すが、使われ方は異なる

### カウントベース
- コーパス（NLPの研究やアプリケーションのために目的をもって収集されたテキストデータ）から自動的にエッセンスを抽出する


```python
#コーパスの準備
text = "You say goodbye and I say hello."

#単語単位に分割
text = text.lower()
text = text.replace(".", " .")

words = text.split(" ")
```


```python
words
```




    ['you', 'say', 'goodbye', 'and', 'i', 'say', 'hello', '.']



- 単語にIDを割り当てる


```python
word_to_id = {}
id_to_word = {}

for word in words:
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word
```


```python
word_to_id
```




    {'you': 0, 'say': 1, 'goodbye': 2, 'and': 3, 'i': 4, 'hello': 5, '.': 6}




```python
id_to_word
```




    {0: 'you', 1: 'say', 2: 'goodbye', 3: 'and', 4: 'i', 5: 'hello', 6: '.'}



- 単語IDリストを作成する


```python
import numpy as np
corpus = [word_to_id[w] for w in words]
corpus = np.array(corpus)
```


```python
corpus
```




    array([0, 1, 2, 3, 4, 1, 5, 6])



- ここまでの前処理を関数にまとめる


```python
def preprocess(text):
    text = text.lower()
    text = text.replace(".", " .")
    words = text.split(" ")
    
    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
            
    corpus = np.array([word_to_id[w] for w in words])
    
    return corpus, word_to_id, id_to_word
```


```python
text = "You say goodbye and I say hello."
preprocess(text)
```




    (array([0, 1, 2, 3, 4, 1, 5, 6]),
     {'you': 0, 'say': 1, 'goodbye': 2, 'and': 3, 'i': 4, 'hello': 5, '.': 6},
     {0: 'you', 1: 'say', 2: 'goodbye', 3: 'and', 4: 'i', 5: 'hello', 6: '.'})



#### 単語の分散表現 
分散表現＝単語の意味をとらえたベクトル

- 色でいえば、青色、空色、瑠璃色、というよりもR,G,Bの数値で表したほうが正確にとらえられる
- この場合、R,G,Bは色をあらわす特徴量で、RGBを用いて色はベクトルで表現できる

#### 分布仮説
分布仮設　＝　単語の意味は周囲の単語によって形成される
- コンテキスト（文脈）によって単語の意味が決定される

言葉の定義
- コンテキスト：注目する単語の前後に存在する単語
- ウィンドウサイズ：コンテキストのサイズ。前後何個の単語を指定するか

#### 共起行列
カウントベースの手法　＝　ある単語の周囲にどのような単語がどれだけ出現するかをカウント集計する


```python
text = "You say goodbye and I say hello."
corpus,word_to_id,id_to_word = preprocess(text)
```


```python
#ウィンドウサイズ＝１
C = np.array([
    [0,1,0,0,0,0,0],
    [1,0,1,0,1,1,0],
    [0,1,0,1,0,0,0],
    [0,0,1,0,1,0,0],
    [0,1,0,1,0,0,0],
    [0,1,0,0,0,0,1],
    [0,0,0,0,0,1,0]
],dtype=np.int32)
```


```python
print(C[0]) #単語IDが0のベクトル
```

    [0 1 0 0 0 0 0]
    


```python
import pandas as pd
co_occurence_matrix = pd.DataFrame(C,index=word_to_id.keys(),columns=word_to_id.keys())
```


```python
co_occurence_matrix
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>you</th>
      <th>say</th>
      <th>goodbye</th>
      <th>and</th>
      <th>i</th>
      <th>hello</th>
      <th>.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>you</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>say</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>goodbye</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>and</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>i</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>hello</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>.</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



- 共起行列作成も関数化する


```python
def create_co_matrix(corpus,vocab_size,window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size,vocab_size),dtype=np.int32)
    
    for idx, word_id in enumerate(corpus):
        for i in range(1,window_size+1):
            left_idx = idx - i
            right_idx = idx + i
            
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
                
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
                
    return co_matrix
```

#### ベクトル間の類似度
単語のベクトル表現の類似度に関しては、コサイン類似度がよくつかわれる
- cosine類似度は-1~1の値をとる


```python
#epsは0除算対策
def cos_similarity(x, y, eps=1e-8):
    nx = x / np.sqrt(np.sum(x**2) + eps) #xの正規化
    ny = y / np.sqrt(np.sum(y**2) + eps) #yの正規化
    return np.dot(nx,ny)
```

- you と i のコサイン類似度を求めてみる


```python
text = "You say goodbye and I say hello."
corpus,word_to_id,id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

c0 = C[word_to_id["you"]] #youの単語ベクトル
c1 = C[word_to_id["i"]] #iの単語ベクトル

print(cos_similarity(c0, c1))
```

    0.7071067758832467
    

#### 類似単語のランキング表示


```python
'''
query : クエリ（単語）
word_to_id : 単語から単語IDへのディクショナリ
id_to_word : 単語IDから単語へのディクショナリ
word_matrix : 単語ベクトルをまとめた行列。各行に対応する単語のベクトルが格納されていることを想定する
top : 上位何位まで表示するか
'''

def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    #クエリを取り出す
    if query not in word_to_id:
        print("%s is not found" % query)
        return 
    
    print("\n[query] " + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]
    
    #コサイン類似度の算出
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)
        
    #コサイン類似度の結果から、その値を高い順に出力
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(" %s: %s" % (id_to_word[i], similarity[i]))
        
        count += 1
        if count >= top:
            return
```

- argsort():numpy配列の要素を小さい順にソートし、そのインデックスを返す


```python
text = "You say goodbye and I say hello."
corpus,word_to_id,id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

most_similar("you", word_to_id, id_to_word, C, top=5)
```

    
    [query] you
     goodbye: 0.7071067758832467
     i: 0.7071067758832467
     hello: 0.7071067758832467
     say: 0.0
     and: 0.0
    

### カウントベース手法の改善

#### 相互情報量
共起行列の要素として、2つの単語が共起した回数を用いたことによる問題点
- 高頻度単語の類似度が高くなってしまう("the"はよく使われる。各々の名詞の類似語で常に"the"が出現 )

↓

相互情報量(Pairwies Mutual Information : PMI)
- 2つの単語の共起する回数が0の場合、log2(0)=-∞となってしまう

↓

正の相互情報量(Positive PMI : PPMI)

PPMI(x,y) = max(0, PMI(x,y) )



```python
def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0
    
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps)
            M[i,j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100) == 0:
                    print("%.1f%% done" % (100+cnt/total))
    return M
        
```


```python
text = "You say goodbye and I say hello."
corpus,word_to_id,id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

np.set_printoptions(precision=3) #有効桁数3桁で表示
print("covariance matrix")
print(C)
print("-" * 50)
print("PPMI")
print(W)
```

    covariance matrix
    [[0 1 0 0 0 0 0]
     [1 0 1 0 1 1 0]
     [0 1 0 1 0 0 0]
     [0 0 1 0 1 0 0]
     [0 1 0 1 0 0 0]
     [0 1 0 0 0 0 1]
     [0 0 0 0 0 1 0]]
    --------------------------------------------------
    PPMI
    [[0.    1.807 0.    0.    0.    0.    0.   ]
     [1.807 0.    0.807 0.    0.807 0.807 0.   ]
     [0.    0.807 0.    1.807 0.    0.    0.   ]
     [0.    0.    1.807 0.    1.807 0.    0.   ]
     [0.    0.807 0.    1.807 0.    0.    0.   ]
     [0.    0.807 0.    0.    0.    0.    2.807]
     [0.    0.    0.    0.    0.    2.807 0.   ]]
    

- 単純なカウントよりも、より単語の性質を表せているベクトルを獲得できた

#### 次元削減

PPMI行列の問題点
- コーパスに含まれる語彙数＝行列の次元数　→　容易に次元が膨らみ、計算が困難
- 行列の中身の多くは0＝ベクトルのほとんどの要素が重要ではない　→　ノイズに弱い

⇓

次元削減
- 重要な情報をできるだけ残したうえで、次元を小さくする

特異値分解
- X = U * S * V.T


```python
text = "You say goodbye and I say hello."
corpus,word_to_id,id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

#SVD
U,S,V = np.linalg.svd(W)

print("共起行列")
print(C[0])
print("PPMI行列")
print(W[0])
print("svd")
print(U[0])
```

    共起行列
    [0 1 0 0 0 0 0]
    PPMI行列
    [0.    1.807 0.    0.    0.    0.    0.   ]
    svd
    [-1.110e-16  3.409e-01 -1.205e-01 -3.886e-16  0.000e+00 -9.323e-01
     -2.087e-17]
    


```python
#2次元ベクトルに削減する
print(U[0,:2])
```

    [-1.110e-16  3.409e-01]
    


```python
#各単語を2次元ベクトルに削減してプロットする
import matplotlib.pyplot as plt
%matplotlib inline

for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
plt.scatter(U[:,0], U[:,1], alpha=0.5)
plt.show()
```


![png](output_43_0.png)


#### PTBデータセット


```python
from dataset import ptb
```


```python
window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data("train")
vocab_size = len(word_to_id)

print("counting co-occurence ...")
C = create_co_matrix(corpus, vocab_size, window_size)

print("calculating PPMI ...")
W = ppmi(C, verbose=True)

print("calculating SVD ...")
try:
    #truncated SVD(fast)
    from sklearn.utils.extmath import randomized_svd
    U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)
except ImportError:
    #SVD(slow)
    U, S, V = np.linalg.svd(W)

word_vecs = U[:, :wordvec_size]

querys = ["you", "year", "car", "toyota"]
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
```

    Downloading ptb.train.txt ... 
    Done
    counting co-occurence ...
    calculating PPMI ...
    

    C:\Users\daich\Anaconda3\lib\site-packages\ipykernel_launcher.py:10: RuntimeWarning: overflow encountered in long_scalars
      # Remove the CWD from sys.path while we load stuff.
    C:\Users\daich\Anaconda3\lib\site-packages\ipykernel_launcher.py:10: RuntimeWarning: invalid value encountered in log2
      # Remove the CWD from sys.path while we load stuff.
    

    100.0% done
    100.0% done
    100.0% done
    100.0% done
    100.0% done
    100.1% done
    100.1% done
    100.1% done
    100.1% done
    100.1% done
    100.1% done
    100.1% done
    100.1% done
    100.1% done
    100.2% done
    100.2% done
    100.2% done
    100.2% done
    100.2% done
    100.2% done
    100.2% done
    100.2% done
    100.2% done
    100.2% done
    100.2% done
    100.3% done
    100.3% done
    100.3% done
    100.3% done
    100.3% done
    100.3% done
    100.3% done
    100.3% done
    100.3% done
    100.3% done
    100.4% done
    100.4% done
    100.4% done
    100.4% done
    100.4% done
    100.4% done
    100.4% done
    100.4% done
    100.4% done
    100.5% done
    100.5% done
    100.5% done
    100.5% done
    100.5% done
    100.5% done
    100.5% done
    100.5% done
    100.5% done
    100.5% done
    100.5% done
    100.6% done
    100.6% done
    100.6% done
    100.6% done
    100.6% done
    100.6% done
    100.6% done
    100.6% done
    100.6% done
    100.7% done
    100.7% done
    100.7% done
    100.7% done
    100.7% done
    100.7% done
    100.7% done
    100.7% done
    100.7% done
    100.7% done
    100.8% done
    100.8% done
    100.8% done
    100.8% done
    100.8% done
    100.8% done
    100.8% done
    100.8% done
    100.8% done
    100.8% done
    100.8% done
    100.9% done
    100.9% done
    100.9% done
    100.9% done
    100.9% done
    100.9% done
    100.9% done
    100.9% done
    100.9% done
    101.0% done
    101.0% done
    101.0% done
    101.0% done
    101.0% done
    101.0% done
    calculating SVD ...
    

    C:\Users\daich\Anaconda3\lib\importlib\_bootstrap.py:219: RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192 from C header, got 216 from PyObject
      return f(*args, **kwds)
    C:\Users\daich\Anaconda3\lib\importlib\_bootstrap.py:219: RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192 from C header, got 216 from PyObject
      return f(*args, **kwds)
    C:\Users\daich\Anaconda3\lib\importlib\_bootstrap.py:219: RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192 from C header, got 216 from PyObject
      return f(*args, **kwds)
    C:\Users\daich\Anaconda3\lib\importlib\_bootstrap.py:219: RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192 from C header, got 216 from PyObject
      return f(*args, **kwds)
    C:\Users\daich\Anaconda3\lib\importlib\_bootstrap.py:219: RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192 from C header, got 216 from PyObject
      return f(*args, **kwds)
    

    
    [query] you
     i: 0.7113931179046631
     we: 0.6759003400802612
     do: 0.5505104064941406
     anybody: 0.5141006708145142
     'll: 0.4904499351978302
    
    [query] year
     month: 0.6773053407669067
     quarter: 0.676358699798584
     next: 0.612481951713562
     fiscal: 0.6090800166130066
     third: 0.5794809460639954
    
    [query] car
     luxury: 0.6458295583724976
     auto: 0.6179211735725403
     truck: 0.5247380137443542
     domestic: 0.49338629841804504
     lexus: 0.48030203580856323
    
    [query] toyota
     motor: 0.7345403432846069
     nissan: 0.6699597239494324
     motors: 0.6484201550483704
     honda: 0.6418188214302063
     mazda: 0.6053842306137085
    


```python

```
