"""所有模型的前向冒烟测试——不需要 GPU、不需要 pickle、不需要 GloVe。

这些测试抓的是最容易犯又最难发现的一类错误：**形状对了但语义错了**。
原始代码里 `permute` 把卷积核维度当时间步、`softmax(dim=1)` 归一化到 batch 维、
`capsule[0]` 索引到 batch 而不是胶囊——它们都不报错，只是训练不出来。

    pytest tests/test_models.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import common  # noqa: E402
from experiments.glove import attention_lstm  # noqa: E402
from experiments.glove import capsule_lstm  # noqa: E402
from experiments.glove import cnn  # noqa: E402
from experiments.glove import cnnlstm  # noqa: E402
from experiments.glove import gru  # noqa: E402
from experiments.glove import lstm  # noqa: E402
from experiments.glove import transformer  # noqa: E402

MODULES = [cnn, lstm, gru, cnnlstm,
           attention_lstm, transformer, capsule_lstm]
VOCAB, EMBED, BATCH, SEQ = 200, 300, 6, 64


@pytest.fixture(scope="module")
def weight() -> torch.Tensor:
    torch.manual_seed(0)
    matrix = torch.randn(VOCAB, EMBED)
    matrix[0] = 0                      # 第 0 行是 <pad>/<unk>
    return matrix


@pytest.fixture(scope="module")
def batch() -> torch.Tensor:
    torch.manual_seed(0)
    inputs = torch.randint(1, VOCAB, (BATCH, SEQ))
    inputs[1, 20:] = 0                 # 中等长度
    inputs[2, 1:] = 0                  # 只有一个词——pack_padded_sequence 的边界情况
    return inputs


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.NAME)
def test_forward_shape_and_finite(module, weight, batch):
    model = module.SentimentNet(weight).eval()
    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (BATCH, 2)
    assert torch.isfinite(logits).all(), "输出出现 nan/inf——通常是 -inf 掩码把整行都掩掉了"


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.NAME)
def test_embedding_is_frozen(module, weight, batch):
    """Embedding 必须冻结：GloVe 是预训练好的，2 万条数据微调它只会过拟合。"""
    model = module.SentimentNet(weight)
    assert not model.embedding.weight.requires_grad


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.NAME)
def test_padding_does_not_change_prediction(module, weight, batch):
    """在真实词后面多补一截 PAD，预测应当基本不变。

    这条测试直接针对原代码 `states[-1]` 的缺陷：把序列填到 512 时，
    「最后一个时间步」读的全是 PAD，句尾信息被冲掉——补的 PAD 越多结果偏移越大。
    RNN 系（LSTM/GRU/Attention/Capsule）和 Transformer 有精确掩码，容差取 1e-3。
    CNN 系放宽：PAD 位置的词向量是零，但卷积仍有 bias，`relu(bias)` 有可能盖过
    真实词的响应而被 max_pool 选中，所以补 PAD 会带来千分之几的扰动。
    CNN-LSTM 在池化之后已经无法还原哪些位置是 PAD，容差最大。
    """
    model = module.SentimentNet(weight).eval()
    short = batch[:, :SEQ]
    padded = torch.cat([short, torch.zeros(BATCH, SEQ, dtype=torch.long)], dim=1)
    with torch.no_grad():
        a = torch.softmax(model(short), dim=1)
        b = torch.softmax(model(padded), dim=1)
    tolerance = {"cnn": 0.02, "cnnlstm": 0.2}.get(module.NAME, 1e-3)
    assert (a - b).abs().max() < tolerance


def test_attention_weights_sum_to_one(weight, batch):
    """注意力权重必须沿**时间轴**归一化。原代码 softmax(dim=1) 归一化到了 batch 维，
    表现就是「每列和为 1」而不是「每行和为 1」，batch_size 一改结果就变。"""
    model = attention_lstm.SentimentNet(weight).eval()
    with torch.no_grad():
        _logits, weights = model.encode(batch)
    assert weights.shape[0] == BATCH
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(BATCH), rtol=1e-4, atol=1e-4)
    # PAD 位置权重必须为 0：第 2 条样本只有 1 个真实词
    assert weights[2, 1:].abs().max() < 1e-6


def test_capsule_squash_keeps_length_below_one():
    """squash 后向量长度必须落在 (0, 1)——长度就是「特征存在的置信度」。
    原代码写的是纯 L2 归一化，所有长度都恰好等于 1，置信度信息被抹平。"""
    tensor = torch.randn(4, 8, 16) * 10
    squashed = capsule_lstm.Capsule.squash(tensor)
    norms = squashed.norm(dim=-1)
    assert (norms > 0).all() and (norms < 1).all()


def test_encode_texts_matches_training_pipeline():
    """新评论的编码必须和训练时完全一致，否则 --predict 的结果没有意义。"""
    word_to_idx = {"<unk>": 0, "great": 1, "movie": 2}
    features = common.encode_texts(["A GREAT movie!!! <br />zzz"], word_to_idx, maxlen=8)
    assert features.shape == (1, 8)
    # a / zzz 不在词表 → 0；great → 1；movie → 2；其余是右侧填充
    assert features[0].tolist() == [0, 1, 2, 0, 0, 0, 0, 0]
