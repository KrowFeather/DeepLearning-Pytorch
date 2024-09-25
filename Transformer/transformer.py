import math
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import copy


# Embedding层，用来进行word2Vec
class Embedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        out = self.embedding(x)
        # 缩放，*根号d_model
        out = out * math.sqrt(self.hidden_dim)
        return out


d_model = 512


# vocab_size = 1000
# x = torch.LongTensor([[1, 2, 3, 4], [2, 6, 7, 9]])
# model = Embedding(vocab_size, d_model)
# embr = model(x)
# print(embr)


# Positional Encoding，由于attention的编码器没有针对其位置信息的处理，通过PE可以获取位置信息
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


# dropout = 0.1
# max_len = 60
# x = embr
# model = PositionalEncoding(d_model, dropout, max_len)
# pe_res = model(x)


# print(pe_res)
# print(pe_res.shape)


# 掩码张量，遮掩一些信息，防止未来信息被提前利用
def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(1 - subsequent_mask)


# size = 5
# sm = subsequent_mask(size=size)
# print(sm)


# attention implementation
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    # print(query.size())
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


# query = key = value = pe_res
# mask = Variable(torch.zeros(2, 4, 4))
# attn, p_attn = attention(query, key, value, mask=mask)
# print('attn', attn)
# print('p_attn', p_attn)


# print(p_attn.shape)

def clones(x, k):
    return nn.ModuleList([copy.deepcopy(x) for _ in range(k)])


# 多头注意力 implementation
class MultiHeadAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert embedding_dim % head == 0
        self.d_k = embedding_dim // head
        self.head = head
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:           mask = mask.unsqueeze(0)
        print(mask.shape)
        batch_size = query.size(0)
        query, key, value = [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2) for model, x in
                             zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_k * self.head)
        x = self.linears[-1](x)
        return x


# head = 8
# embedding_dim = 512
# dropout = 0.2
# query = key = value = pe_res
# mask = Variable(torch.zeros(8, 4, 4))
# mha = MultiHeadAttention(head, embedding_dim, dropout)
# mha_result = mha(query, key, value, mask)
# print(mha_result)

# 前馈全连接子层
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# d_ff = 64
# dropout = 0.2
# x = mha_result
# ff = PositionwiseFeedForward(d_model, d_ff, dropout)
# ff_res = ff(x)
# print(ff_res)
# print(ff_res.shape)

# LayerNorm层
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# features = d_model = 512
eps = 1e-6


# x = ff_res
# ln = LayerNorm(features, eps=eps)
# ln_res = ln(x)
# print(ln_res)
# print(ln_res.shape)

# 用来将ff或attn与norm，dropout结合
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# size = d_model = 512
# head = 8
# dropout = 0.2
# x = pe_res
# mask = Variable(torch.zeros(8, 4, 4))
# self_attn = MultiHeadAttention(head, d_model)
# sublayer = lambda x: self_attn(x, x, x, mask)
# sc = SublayerConnection(size, dropout)
# sc_res = sc(x, sublayer)
# print(sc_res)
# print(sc_res.shape)

# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, size, dropout, self_attn, ff):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.ff = ff
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda y: self.self_attn(y, y, y, mask))
        x = self.sublayer[1](x, self.ff)
        return x


# el_res = el(x, mask)
# print(el_res)
# print(el_res.shape)

# 编码器：编码器层*N + LayerNorm
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


size = 512
dropout = 0.2
d_model = 512
d_ff = 64
head = 8
# mask = Variable(torch.zeros(8, 4, 4))
self_attn = MultiHeadAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
# x = pe_res
c = copy.deepcopy
el = EncoderLayer(d_model, dropout, c(self_attn), c(ff))
N = 5
en = Encoder(el, N)


# en_res = en(x, mask)


# print(en_res)
# print(en_res.shape)

# 解码器层
class DecoderLayer(nn.Module):
    def __init__(self, size, src_attn, self_attn, ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.ff = ff
        self.src_attn = src_attn
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        x = self.sublayer[0](x, lambda y: self.self_attn(y, y, y, target_mask))
        x = self.sublayer[1](x, lambda y: self.src_attn(y, memory, memory, source_mask))
        x = self.sublayer[2](x, self.ff)
        return x


size = d_model = 512
head = 8
d_ff = 64
dropout = 0.2
self_attn = MultiHeadAttention(head, d_model, dropout)
src_attn = MultiHeadAttention(head, d_model, dropout)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
# x = pe_res
# memory = en_res
c = copy.deepcopy
source_mask = target_mask = Variable(torch.ones(8, 4, 4))
dl = DecoderLayer(size, c(src_attn), c(self_attn), c(ff), dropout)


# dl_res =  dl(x, memory, source_mask, target_mask)
# print(dl_res)
# print(dl_res.shape)

# 解码器：解码器层*N + layernorm
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(d_model, eps=1e-6)

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


N = 5
de = Decoder(dl, N)


# de_res = de(x, memory, source_mask, target_mask)


# print(de_res)
# print(de_res.shape)

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.project(x), dim=-1)


# x = de_res
# gen = Generator(d_model, vocab_size)
# gen_res = gen(x)
# print(gen_res)
# print(gen_res.shape)

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, trg_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        return self.generator(self.decode(self.encode(source, source_mask), source_mask, target, target_mask))

    def encode(self, source, source_mask):
        return self.encoder(self.src_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
        return self.decoder(self.trg_embed(target), memory, source_mask, target_mask)


vocab_size = 1000
d_model = 512
encoder = en
decoder = de
source_embed = nn.Embedding(vocab_size, d_model)
target_embed = nn.Embedding(vocab_size, d_model)
gen = Generator(d_model, vocab_size)

source = target = Variable(torch.LongTensor([[100, 2, 421, 500], [491, 998, 1, 221]]))
source_mask = target_mask = Variable(torch.ones(8, 4, 4))
ed = EncoderDecoder(encoder, decoder, source_embed, target_embed, gen)
ed_result = ed(source, target, source_mask, target_mask)
print(ed_result)
print(ed_result.shape)


def make_model(source_vocab, target_vocab, N=6, d_model=512, d_ff=2048, head=8, dropout=0.2):
    c = copy.deepcopy
    attn = MultiHeadAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, dropout,c(attn), c(ff)), N),
        Decoder(DecoderLayer(d_model,c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embedding(d_model,source_vocab),c(position)),
        nn.Sequential(Embedding(d_model,target_vocab),c(position)),
        Generator(d_model, target_vocab)
    )

    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)

    return model

if __name__ == '__main__':
    source_vocab = 11
    target_vocab = 11
    N = 6
    model = make_model(source_vocab, target_vocab,N)
    print(model)