
class Embedder(nn.Module):
    def __init__(self, text_embedding_vectors):
        super().__init__()

        self.embeddings = nn.Embedding.from_pretrained(
            embeddings=text_embedding_vectors, 
            freeze=True
        )

    def forward(self, x):
        x_vec = self.embeddings(x)
        return x_vec

class PositionalEncoder(nn.Module):
    def __init__(self, d_model=300, max_seq_len=256):
        super().__init__() 

        self.d_model = d_model 
        pe = torch.zeros(max_seq_len, d_model)

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2*i)/d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i+1))/d_model)))

        self.pe = pe.unsqueeze(0)
        self.pe.requires_grad = False 
    
    def forward(self, x):
        ret = math.sqrt(self.d_model) * x + self.pe 
        return ret 

class Attention(nn.Module):
    def __init__(self, d_model=300):
        super().__init__() 

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model) 

        self.out = nn.Linear(d_model, d_model)

        self.d_k = d_model

    def forward(self, q, k, v, mask):
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)

        # query * key.T
        weights = torch.matmul(q, k.transpose(1, 2) / math.sqrt(self.d_k))
        # <pad> の箇所を -∞ に置き換える
        mask = mask.unsqueeze(1)
        weights = weights.masked_fill(mask==0, -1e9)
        # softmax で 0~1 に収める。softmax(-inf) = 0 なので、<pad> の箇所の重みは 0 になる
        normalized_weights = F.softmax(weights, dim=-1)

        # weights + value
        output = torch.matmul(normalized_weights, v)
        output = self.out(output)

        return output, normalized_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout(F.relu(x))
        x = self.linear_2(x)
        return x 

class TransformerBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        # Layer Normalization
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        # Attention
        self.attn = Attention(d_model)

        # FC * 2
        self.ff = FeedForward(d_model)

        # Dropout
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x_normalized = self.norm_1(x)
        output, normalized_weights = self.attn(
            x_normalized, x_normalized, x_normalized, mask
        )
        x2 = x + self.dropout_1(output)

        x_normalized2 = self.norm_2(x2)
        output = x2 + self.dropout_2(self.ff(x_normalized2))

        return output, normalized_weights

class ClassificationHead(nn.Module):
    def __init__(self, d_model=300, output_dim=2):
        super().__init__()

        self.linear = nn.Linear(d_model, output_dim)

        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)
    
    def forward(self, x):
        x0 = x[:, 0, :] # 各ミニバッチの先頭単語の特徴量（300次元） = <cls> トークン
        out = self.linear(x0)
        return out 

class TransformerClassification(nn.Module):
    def __init__(self, text_embedding_vectors, d_model=300, max_seq_len=256, output_dim=2):
        super().__init__()

        self.net1 = Embedder(text_embedding_vectors)
        self.net2 = PositionalEncoder(d_model=d_model, max_seq_len=max_seq_len)
        self.net3_1 = TransformerBlock(d_model=d_model)
        self.net3_2 = TransformerBlock(d_model=d_model)
        self.net4 = ClassificationHead(output_dim=output_dim, d_model=d_model)

    def forward(self, x, mask):
        x1 = self.net1(x)
        x2 = self.net2(x)
        x3_1, normalized_weights_1 = self.net3_1(x2, mask)
        x3_2, normalized_weights_2 = self.net3_2(x3_1, mask)
        x4 = self.net4(x3_2)
        return x4, normalized_weights_1, normalized_weights_2
