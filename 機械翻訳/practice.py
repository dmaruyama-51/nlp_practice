import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

data = [
    [1, 2, 3, 4],
    [5, 6, 7, 0],
    [8, 9, 0, 0]
]

len_sequences = [4, 3, 2] # 実際の seq_length

x = torch.LongTensor(data)
x = x.t() # (batch_size, seq_length) -> (seq_length, batch_size)

h = pack_padded_sequence(x, len_sequences)
print(h)

'''
PackedSequence(
    data=tensor([1, 5, 8, 2, 6, 9, 3, 7, 4]), 
    batch_sizes=tensor([3, 3, 2, 1]), 
    sorted_indices=None, 
    unsorted_indices=None
    )

data : 元のテンソルからパディング部分を覗いた値のみを保持したテンソル
batch_sizes : 各時刻において計算が必要なデータ数を表すテンソル
'''

y, _ = pad_packed_sequence(h)
print(y)