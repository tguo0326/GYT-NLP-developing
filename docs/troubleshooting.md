# 遇到的问题及解决办法

按「环境 → 数据 → 词向量 → 代码 → 训练」的顺序记录。每条都是实际踩到的，
不是照抄常见问题清单。

## 一、原始代码里跑不起来的问题

### 1. 写死的 Windows 绝对路径

```python
# imdb_process.py 原文
wvmodel_file = os.path.join("g:\\", 'lib', 'glove.840B.300d.gensim.txt')
```

Linux / macOS 上直接 `FileNotFoundError`。改成相对路径 `glove/`，
并加了明确的报错提示（告诉你该跑哪个脚本去生成）。

### 2. `pickle/` 目录不存在

原代码直接 `open(os.path.join('pickle', 'imdb_glove.pickle3'), 'wb')`。
预处理跑了几分钟，到最后一行才因为目录不存在崩掉，前面的工作全白费。
现在 `args.output.parent.mkdir(parents=True, exist_ok=True)`。

同类问题：所有模型脚本都往 `./result/cnn.csv` 写（注意是单数 `result`），
目录同样不存在，而且是在**训练完 10 个 epoch 之后**才崩。统一改到 `results/`
并由 `common.setup()` 提前创建。

### 3. 文件名不一致

`imdb_process.py` 产出 `pickle/imdb_glove.pickle3`，
但 `imdb_cnn.py` 读的是 `pickle/imdb_demo_glove.pickle3`。统一成前者。

### 4. `device = torch.device('cuda:0')` + `.cuda()` 写死

没有 GPU 的机器上第一个 batch 就 `AssertionError: Torch not compiled with CUDA`。
改成 `common.get_device()` 自动选择 CUDA / MPS / CPU，
所有 `.cuda()` 换成 `.to(device)`。

### 5. `train_loss += loss` 累加的是张量

```python
train_loss += loss          # loss 是带计算图的 Tensor
```

每个 batch 的计算图都被这个引用挂住不释放，一个 epoch 下来显存单调增长，
长序列 + 大 batch 时必然 OOM。改成 `total_loss += loss.item() * batch_size`。

### 6. 没有 `model.train()` / `model.eval()`

原代码全程处于默认的 train 模式。后果是**验证时 Dropout 仍然随机丢神经元、
BatchNorm 仍在更新统计量**，验证准确率被系统性低估且每次不同。
现在由 `common._run_epoch()` 按是否传入 optimizer 自动切换。

### 7. 没有固定随机种子

每次跑出来的数字都不一样，无法判断某个改动到底有没有用。
`common.set_seed(42)` 统一固定 `random` / `numpy` / `torch` / CUDA，
并设 `cudnn.deterministic = True`。

### 8. 不保存模型

原代码跑完 10 个 epoch 直接用**最后一个** epoch 的权重预测测试集。
但 IMDB 上这些模型普遍在第 4~7 个 epoch 就到最好，之后过拟合
（实测 CNN：epoch 5 的 `val_acc` 0.8876，epoch 10 掉到 0.8850，
`val_loss` 从 0.29 涨到 0.35）。现在按 `val_acc` 保存最佳权重，
训练结束自动载回最佳那份再做预测。

## 二、维度与语义层面的错误（不报错，但训不出来）

这一类最难发现——代码跑得通、loss 在降、只是降不到该有的水平。

### 9. 序列填充到 512，`states[-1]` 读的全是 PAD

```python
states, hidden = self.encoder(embeddings.permute([1, 0, 2]))
encoding = torch.cat([states[0], states[-1]], dim=1)
```

评论长度中位数是 177 词，却被填充到 512。正向 LSTM 的「最后一个时间步」
实际读完了三百多个 PAD，真实句尾的信息早被冲掉。
改用 `pack_padded_sequence`，LSTM 只跑有效长度，`h_n` 就是真正的句尾状态。

`tests/test_models.py::test_padding_does_not_change_prediction` 就是钉这条的：
在真实词后面多补一截 PAD，预测不应该改变。

### 10. CNN-LSTM 把卷积核维度当成了时间步

```python
pooling = F.max_pool1d(convolution, kernel_size=2)   # (batch, 128, 256)
states, hidden = self.encoder(pooling.permute([1, 0, 2]))   # (128, batch, 256)
self.encoder = nn.LSTM(input_size=max_len // pooling_size)  # input_size=256
```

`permute([1, 0, 2])` 之后，LSTM 认为序列长度是 **128（卷积核个数）**、
特征维度是 **256（时间步）**。也就是在一个无序的卷积核集合上做顺序建模，
而真正的词序被塞进了特征向量里——组合模型的意义完全没有了。
正确形状是 `(batch, seq', num_filter)`，`input_size=num_filter`。

### 11. Attention 的 softmax 归一化到了 batch 维

```python
att_score = F.softmax(att, dim=1)     # states 是 (seq_len, batch, hidden)
outputs = x * att_score               # 而且没有求和
```

`states` 没用 `batch_first`，所以 `dim=1` 是 batch 维。归一化跑到了
「同一个 batch 里的不同样本」之间——**batch_size 一改结果就变**，
而且这件事本身毫无意义。其次 `x * att_score` 之后没有 `sum`，
Attention 层返回的还是整条序列，紧接着又走 `torch.cat([states[0], states[-1]])`，
加权算完就被丢掉了。

修好之后注意力权重立刻变得可解释（`results/attention_lstm_attention.json`）：

```
wonderful(0.167), absolutely(0.150), superb(0.130), acting(0.092)   → 预测正面
waste(0.124), terrible(0.117), of(0.104), time(0.102), wooden(0.058) → 预测负面
```

`tests/test_models.py::test_attention_weights_sum_to_one` 钉住「每行和为 1」
以及「PAD 位置权重为 0」。

### 12. Capsule 的 squash 写成了 L2 归一化

```python
scale = torch.sqrt(s_squared_norm + 1e-7)
return x / scale                       # 所有胶囊长度都变成 1
```

胶囊网络的关键设定是「向量长度 = 特征存在的置信度」。纯 L2 归一化把所有长度
都变成 1，置信度这一路信息被完全抹平。正确的 squash 是
`‖s‖²/(1+‖s‖²) · s/‖s‖`。另外原代码 `capsule[0]` / `capsule[-1]`
索引的是 batch 维（形状是 `(batch, num_capsule, dim_capsule)`），
取到的是「batch 里的第一个和最后一个样本」。

### 13. Transformer：逐字符建词表

```python
def review_to_wordlist(review):
    ...
    return ' '.join(words)              # 返回字符串，不是列表

class Vocab:
    def build(cls, train, test):
        for sentence in train:
            for token in sentence:      # 遍历字符串 = 遍历字符
```

词表退化成 26 个字母加空格。同一个文件里还有：`d_model=120` 但输入是 300 维、
`F.log_softmax` 的输出又喂给 `CrossEntropyLoss`（等于取两次对数）、
验证循环 `net(val_feature)` 少传 `lengths` 参数、
`train_test_split` 之后 `train_labels` 被覆盖导致用了两套标签。
这个脚本重写了。

## 三、新版库的 API 变更

| 旧写法 | 新写法 | 变更版本 |
|---|---|---|
| `gensim.models.KeyedVectors.load_word2vec_format(txt)` | `KeyedVectors.load(kv, mmap='r')` | gensim 4.x（也可用 `no_header=True` 读原始 txt） |
| `wvmodel.index2word` / `.vocab` | `.index_to_key` / `.key_to_index` | gensim 4.0 |
| `datasets.load_metric("accuracy")` | 用 `evaluate` 包，或直接 numpy 算 | datasets 2.x 移除 |
| `TrainingArguments(evaluation_strategy=)` | `eval_strategy=` | transformers 4.46 |
| `Trainer(tokenizer=)` | `Trainer(processing_class=)` | transformers 4.46 弃用 |
| `torch.autograd.Variable(x)` | 直接用 `x` | PyTorch 0.4 起 Tensor 自带 autograd |

## 四、GloVe 相关的坑

### 14. Gensim 读不了原始的 `glove.840B.300d.txt`

两个原因叠加：

1. GloVe 的 txt 没有 `<词数> <维度>` 头部行（`no_header=True` 能解决）；
2. 840B 这一份里有若干「词」本身包含空格。Gensim 按空格切分后要求恰好 301 段，
   遇到这些行直接 `ValueError`。

`tools/prepare_glove.py` 自己解析：**从右边取 300 个数当向量，剩下的整段当词**。
实测 2,196,017 行全部解析成功，0 行异常。

### 15. `KeyedVectors(vector_size=300, count=N)` 会让词表变成两倍

`count` 是**预分配**语义。之后再调 `add_vectors(words, matrix)`，
新词会追加在预分配的空行**之后**，词表变成 4,392,034 个（一半是没有 key 的零向量），
`.npy` 从 2.5 GB 涨到 5.0 GB，`most_similar` 还会因为对零向量求余弦而刷
`RuntimeWarning: invalid value encountered in divide`。
去掉 `count` 参数即可。

### 16. mmap 打开的 `.npy` 是只读的

```
UserWarning: The given NumPy array is not writable, and PyTorch does not
support non-writable tensors.
```

`torch.from_numpy(np.asarray(vectors[word]))` 直接包了只读数组。
改成 `np.array(vectors[word], dtype=np.float32)` 显式复制。

### 17. 磁盘占用

`glove.840B.300d.zip` 2.1 GB，解压后 txt 5.3 GB，转成 Gensim 原生格式再加 2.5 GB。
**准备 GloVe 至少要留 10 GB 空余磁盘。** 转换完成后 zip 和 txt 都可以删，
只保留 `.kv` + `.kv.vectors.npy`（约 2.5 GB）。

## 五、数据与评测

### 18. `read_csv` 必须显式 `quoting=csv.QUOTE_NONE`

影评正文里全是英文引号。按默认规则解析，字段会被错误合并——
**行数看起来正常，内容已经错位**。原代码用的 `quoting=3` 就是
`csv.QUOTE_NONE`，这一点原作者是对的，迁移时不要「顺手清理」掉。

### 19. 划分不分层、不固定随机数

原代码 `train_test_split(..., test_size=0.2, random_state=0)` 没有 `stratify`。
更麻烦的是三个 BERT 脚本里连 `random_state` 都没有，每次划分都不同——
跨模型的准确率完全不可比。现在全项目统一：
`test_size=0.2, random_state=42, stratify=y`，
`imdb_bow_baseline.py` 和 `hf_trainer.py` 也用同一套，所以对比表里的数字可以直接比较。

### 20. OOV 率看词型会严重高估

按词型算，22.23%（22,106 / 99,420）的词没有 GloVe 向量，看起来很糟。
按词次算只有 **0.43%**——OOV 几乎全是只出现一两次的拟声词
（`aaaaaaaargh`）、拼写错误，和清洗残留的碎片（`isn't` → `isn`）。
**报 OOV 一定要报词次口径。**

### 21. 提交文件用全量数据重训

验证集只用于选型。生成 Kaggle 提交时，传统分类器用全部 25,000 条标注数据重训
（`imdb_bow_baseline.py` 里的 `full_model`）；
神经网络则载回验证集上的最佳权重再预测——这两种做法都是标准的，
但要清楚自己在做哪一种。

## 六、跑起来之后的实用问题

### 22. tqdm 进度条把日志刷爆

重定向到文件时，tqdm 的每次回车刷新都会变成新的一行。跑一次 CNN 产生了
**458 KB** 的日志，全是进度条残片。
`tqdm(..., disable=not sys.stderr.isatty())`：非交互式运行就关掉进度条，
真正的指标每个 epoch 由 `logging` 记录一次。

### 23. 学习率照抄原代码不会收敛

原代码 CNN 和 GRU 用 `SGD(lr=0.8)`、LSTM 用 `Adam(lr=0.05)`。
`Adam` 配 0.05 在 LSTM 上第一个 epoch 就发散到 0.5 准确率（等于瞎猜）。
统一改成 `Adam(lr=1e-3)`；Transformer 更敏感，用 `AdamW(lr=3e-4)`；
RoBERTa 微调用 `1e-5`（5e-5 常见的表现是 loss 卡在 0.69、塌成全预测同一类）。

### 24. HF Trainer 的 `output_dir='./results'` 会污染结果目录

Trainer 往 `output_dir` 写完整的 checkpoint（每份几百 MB）。
原代码指到 `./results`，正好是我们存对比表的目录。改到 `models/<name>_hf_ckpt/`，
并在训练结束后删掉中间 checkpoint，只保留最终模型。
