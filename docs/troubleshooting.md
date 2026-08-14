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

`experiments/preprocess.py` 产出 `pickle/imdb_glove.pickle3`，
但 `experiments/glove/cnn.py` 读的是 `pickle/imdb_demo_glove.pickle3`。统一成前者。

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
`experiments/baseline.py` 和 `core/hf_trainer.py` 也用同一套，所以对比表里的数字可以直接比较。

### 20. OOV 率看词型会严重高估

按词型算，22.23%（22,106 / 99,420）的词没有 GloVe 向量，看起来很糟。
按词次算只有 **0.43%**——OOV 几乎全是只出现一两次的拟声词
（`aaaaaaaargh`）、拼写错误，和清洗残留的碎片（`isn't` → `isn`）。
**报 OOV 一定要报词次口径。**

### 21. 提交文件用全量数据重训

验证集只用于选型。生成 Kaggle 提交时，传统分类器用全部 25,000 条标注数据重训
（`experiments/baseline.py` 里的 `full_model`）；
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

## 七、阶段三：参数高效微调（PEFT）

`new work/` 里那四份 demo 的老坑和阶段二一模一样（`./result/` 目录不存在、
`evaluation_strategy` 已改名、`train_test_split` 缺 `random_state`），
不再重复。这里只记这一阶段**新**踩到的。
完整原理与显存实测见 [peft-lora.md](peft-lora.md)。

### 25. `ValueError: Attempting to unscale FP16 gradients`

为了省显存把底座用 `torch_dtype=float16` 加载，结果 LoRA 参数也跟着变成 fp16，
而 AMP 的 `GradScaler` 明确拒绝 unscale fp16 梯度——混合精度的 master weight
必须是 fp32。

这里有个容易混淆的点：**`fp16=True` 和 fp16 权重不是一回事**。
前者是"计算时转半精度"，activation 已经省了一半，权重仍是 fp32；
后者才是把权重本身存成半精度。

默认只开前者。确实需要 fp16 权重（比如 15.7 亿参数的 xxlarge）时，
把可训练的那 0.x% 单独转回 fp32：

```python
for param in model.parameters():
    if param.requires_grad:
        param.data = param.data.float()
```

这也是 peft 官方 `prepare_model_for_kbit_training` 做的同一件事。

### 26. `element 0 of tensors does not require grad`（LoRA + gradient checkpointing）

最容易踩、也最难猜的一个。PEFT 把底座整个冻结后，第一层的输入
`requires_grad=False`，checkpoint 段会判断"这段不需要梯度"而**根本不建计算图**，
反向传播时自然找不到 `grad_fn`。

```python
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()      # ← 少了这行就报上面的错
```

`use_reentrant=False` 也不是可选的：reentrant 实现和冻结参数、
和 Trainer 对 `use_cache` 的处理都容易打架，PyTorch 2.x 明确推荐非 reentrant。

### 27. Prefix-Tuning 在 DeBERTa 上根本跑不了

两个报错，同一个根因：

```
ValueError: PREFIX_TUNING does not work with gradient checkpointing.
ValueError: Model does not support past key values which are
            required for prefix tuning.
```

peft 把前缀伪装成 `past_key_values`（KV cache）注入每一层，这要求底座实现
KV cache。DeBERTa 是纯 encoder，`DebertaV2Model` 没有这个入口。
gradient checkpointing 那条报错也是同理——反向重算时那份缓存不在图里了。

不是配置问题，改不出来。Prefix-Tuning 换到 `roberta-base`（支持 `past_key_values`）
作为补充行。**结论本身有价值：LoRA 系几乎通吃（有线性层就能挂），
prompt 系挑架构。** 这是工程上默认选 LoRA 的一个隐性理由。

### 28. AdaLoRA 缺 `total_step` 直接 ValueError

peft 0.15 起 `AdaLoraConfig` 必填 `total_step`——秩的裁剪计划
（`tinit` 预热 → 每 `deltaT` 步裁一次 → `tfinal` 固定）是按总步数排的。
demo 写于旧版本，没有这个参数。用 `epochs × steps_per_epoch` 算出来传进去；
探测模式只跑几步时 `tinit`/`tfinal` 会被算成 0，要兜 `max(1, ...)`。

### 29. `target_modules=['q_proj','v_proj']` 在 DeBERTa 上找不到模块

demo 注释里的这两个名字是 LLaMA 的命名。DeBERTa 叫 `query_proj` / `value_proj`。
**最稳的做法是留空**，让 peft 用它按模型类型维护的默认映射——
比手写名字更不容易随底座变化而失效。

### 30. 沿用全量微调的学习率会几乎学不动

PEFT 只训 0.3% 的参数，梯度信号比全模型微调弱得多。
照抄阶段二的 `2e-5` 表现是 loss 降得极慢、跑完两个 epoch 还在 0.6 左右。
提到 `1e-4`（高一个数量级）才正常收敛。

### 31. 单卡试错会把整机拖死

比 CUDA OOM 危险的是**主机内存**被吃光触发 swap：CUDA 的 OOM 是可以
`try/except` 的普通异常，而进了 swap 连 Ctrl-C 都按不进去。
`from_pretrained` 加载 15.7 亿参数的 fp32 权重要 6 GB，加上 safetensors
的临时拷贝可能翻倍。

`core/mem_guard.py` 做三层保险：`set_per_process_memory_fraction` 给显存设硬上限
（超限抛可捕获的 CUDA OOM，而不是吃满整张卡害死同卡的其他进程）、
Trainer 回调每 10 步查一次进程 RSS（PyTorch 管不到主机内存）、
被中止的运行**不写 summary**——绝不让半截的训练冒充正常实验混进对比表。

上限设 13.5 GB / 15.36 GB：留约 1.5 GB 给 CUDA context、cuDNN workspace 和碎片。
实测顶到 14.5 GB 以上时，失败方式会从"干净的异常"退化成驱动层报错、进程 kill 不掉。

### 32. 正式跑之前一定要先探显存

正式训练一次要一小时，显存不够可能在第 40 分钟才爆——白等，还可能拖死机器。
`--probe-steps 20` 只跑 20 步、不做验证不落盘，几十秒读一次峰值就退出。
**显存峰值在前十几步就稳定了，之后基本是平的**，所以 20 步足够。

### 33. AdaLoRA 静默训不出来：三个坑叠在一起

第一次跑出来验证准确率 **0.5094**（二分类瞎猜就是 0.5），loss 稳稳停在
0.6931 = ln2。三个原因，**一个都不报错**：

**① 默认挂载点比 LoRA 宽 6 倍。** `target_modules` 留空时 peft 给两者的默认映射不同：

```
LoRA     {query_proj, value_proj}                    48 处   157 万参数
AdaLoRA  {query_proj, key_proj, value_proj, dense}  290 处  1422 万参数
```

`dense` 把所有 FFN 层都圈进来了。这样比出来的不是「同样预算会不会分配」，
而是「谁的预算大」，实验本身就失效了。AdaLoRA 必须显式写死
`target_modules=["query_proj", "value_proj"]`。

**② `update_and_allocate()` 必须每步手动调，HF Trainer 不会调。**
peft 的 docstring 明确要求「after `loss.backward()` and before `zero_grad()`」。
不调的后果：rank 裁剪从未发生，而 `AdaLoraModel.forward` 里的正交正则项
（`orth_reg_weight` 默认 0.5）一直加在 loss 上——起始 loss 是 2.34 而不是 0.693，
模型把容量全用来压这一项。用 `on_pre_optimizer_step` 回调挂上即可。

**③ 学习率不能沿用 LoRA 的。** AdaLoRA 的增量被除以 ranknum
（`scaling / (ranknum + 1e-5)`，`init_r=32`），等效步长小一个数量级以上。
LoRA 用 1e-4，AdaLoRA 要 5e-4。

修完三处：**0.5094 → 0.9384**。

**附带一个会让人误判的现象**：AdaLoRA 的第一个 epoch 看起来是死的
（`val_acc 0.5256`），第二个 epoch 才跳到 0.9384——前期容量都花在压正交正则和
调 rank 分配上。**别看到第一个 epoch 0.52 就掐掉**，这和 LoRA 第一个 epoch
就有 0.955 是完全不同的形态。

### 34. 把「训练不足」当成「方法不行」

Prefix-Tuning 按和其他方法一样的 2 epochs 跑，只有 0.8546，看起来就是这个方法弱。
但看 loss：结束时还在 **0.41**，而 P-Tuning 是 **0.138**——曲线明显还在降。

加到 6 epochs + lr 3e-4 后是 **0.8976**（+4.3 个百分点），
而且第 5、6 轮是 0.8976 / 0.8974，**曲线压平了才算真收敛**。

教训：跨方法对比时「统一 epochs」不等于「统一收敛程度」。
prompt 类方法（Prefix / P-Tuning）可训练参数极少、梯度信号弱，
需要的 epochs 本来就比 LoRA 多。下结论前先看 loss 有没有平——
**别用一条还在下降的曲线去代表一个方法的上限**。
