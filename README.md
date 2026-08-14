# IMDB 情感分类：从词频统计到 LoRA

同一份 IMDB 影评数据（25,000 条标注 / 25,000 条测试），同一套划分和评测口径，
跑通 16 种做法：词频统计 → 静态词向量 + 各种网络 → 预训练模型全量微调 → 参数高效微调。

想回答的问题是：准确率是靠什么涨上去的，代价是多少参数和多少时间。

## 结果

指标全部是**测试集**：官方 `testData.tsv` 的 25,000 条，其中 24,961 条能与 Stanford
aclImdb 的公开标签按正文对齐，指标在这些行上算。验证集只用来选 epoch，不进表。
单卡 Tesla T4。

<!-- RESULTS_TABLE_START -->

| 模型 | 文本表示 | 测试集准确率 | ROC-AUC | 可训练参数 | 训练时间 |
| --- | --- | --: | --: | --: | --: |
| LoRA | DeBERTa-v3-large，冻结底座 | 0.9633 | 0.9924 | 2,624,514 | 3148 s |
| P-Tuning | DeBERTa-v3-large，冻结底座 | 0.9537 | 0.9906 | 302,338 | 3250 s |
| AdaLoRA | DeBERTa-v3-large，冻结底座 | 0.9607 | 0.9904 | 4,198,914 | 3350 s |
| RoBERTa | 全量微调 | 0.9369 | 0.9834 | 124,647,170 | 555 s |
| BERT | 全量微调 | 0.9174 | 0.9746 | 109,483,778 | 544 s |
| DistilBERT | 全量微调 | 0.9094 | 0.9703 | 66,955,010 | 242 s |
| 逻辑回归 | TF-IDF（1-2gram, 200,000） | 0.9055 | 0.9666 | 200,000 | 16 s |
| Attention-LSTM | GloVe 840B.300d | 0.9027 | 0.9650 | 902,402 | 305 s |
| Capsule-LSTM | GloVe 840B.300d | 0.9036 | 0.9648 | 868,610 | 250 s |
| CNN-LSTM | GloVe 840B.300d | 0.8998 | 0.9637 | 314,498 | 64 s |
| BiGRU | GloVe 840B.300d | 0.8985 | 0.9630 | 565,442 | 283 s |
| BiLSTM | GloVe 840B.300d | 0.8975 | 0.9629 | 753,602 | 293 s |
| Prefix-Tuning | RoBERTa-base，冻结底座 | 0.8964 | 0.9608 | 960,770 | 1083 s |
| TextCNN | GloVe 840B.300d | 0.8858 | 0.9579 | 461,954 | 98 s |
| Transformer | GloVe 840B.300d | 0.8815 | 0.9541 | 1,342,026 | 701 s |
| 随机森林 | Bag of Words（5,000 词频） | 0.8419 | 0.9204 | 5,000 | 12 s |

- 最好的是 **LoRA**：准确率 0.9633，AUC 0.9924
- 比最好的全量微调（RoBERTa 0.9369）高 2.64 个百分点，可训练参数只有它的 1/47
- 最省的是 **P-Tuning**：302,338 个参数（0.07%）拿到 0.9537

<!-- RESULTS_TABLE_END -->

三条结论：

1. 准确率的台阶来自文本表示，不是网络结构。0.84（词频）→ 0.90（GloVe）→
   0.93（BERT 系上下文词向量）→ 0.96（更大的底座）。同一级里换结构只在 1 个百分点内浮动，
   七个 GloVe 模型全部落在 0.88~0.90。
2. TF-IDF 加逻辑回归 0.9055，16 秒，比七个 GloVe 神经网络里的六个都高。
   上手新任务先把这条基线跑出来。
3. 参数高效微调不是省钱的妥协，是让小卡用得上大底座的手段。全量微调 4.37 亿参数的
   DeBERTa-v3-large，光权重加 Adam 状态就要 7 GB，T4 上再加 activation 基本没戏；
   挂上 LoRA 只训 0.6%，峰值 4.8 GB。

### 参数高效微调

<!-- PEFT_TABLE_START -->

| 方法 | 测试集准确率 | ROC-AUC | 可训练参数 | 占全模型 | 显存峰值 | adapter |
| --- | --: | --: | --: | --: | --: | --: |
| LoRA | 0.9633 | 0.9924 | 2,624,514 | 0.60% | 4.80 GB | 21 MB |
| P-Tuning | 0.9537 | 0.9906 | 302,338 | 0.07% | 4.93 GB | 11 MB |
| AdaLoRA | 0.9607 | 0.9904 | 4,198,914 | 0.96% | 4.87 GB | 21 MB |
| Prefix-Tuning | 0.8964 | 0.9608 | 960,770 | 0.76% | 3.96 GB | 8 MB |

<!-- PEFT_TABLE_END -->

Prefix-Tuning 那行的底座是 RoBERTa-base 而不是 DeBERTa，所以不和前三行比准确率。
它在 DeBERTa 上跑不了：peft 通过 `past_key_values` 注入前缀，而 DeBERTa 是纯 encoder，
没有 KV cache。

显存那一栏是这一阶段最实在的收获。同样三招（冻结底座、gradient checkpointing、
小 batch 配梯度累积），15.7 亿参数的 `deberta-v2-xxlarge` 在 T4 上峰值只要 3.49 GB，
而全量微调它需要约 25 GB。没拿它做主实验是因为一个 epoch 要 1.9 小时。
原理和实测数据见 [docs/peft-lora.md](docs/peft-lora.md)。

## 提交文件

`submissions/` 下每个方法一个子目录，各含 `submission.csv`、`summary.json` 和说明。
竞赛指标是 ROC-AUC，所以交的是正面情感概率而不是 0/1 标签，交硬标签会差 3~5 个点。
最好的一份是 `submissions/16_lora/submission.csv`。

## 目录结构

```
core/                共享实现
  common.py            数据加载、训练循环、设备与种子、日志、提交文件
  hf_trainer.py        BERT 系全量微调
  peft_trainer.py      LoRA / AdaLoRA / P-Tuning / Prefix-Tuning
  mem_guard.py         显存与内存看门狗，越线主动中止
experiments/         运行入口
  preprocess.py        清洗、建词表、GloVe 初始化，产出 pickle
  baseline.py          Bag of Words 与 TF-IDF
  glove/               七个 GloVe 神经网络
  finetune/            BERT / DistilBERT / RoBERTa
  peft/                lora / adalora / ptuning / prefix
tools/               数据体检、GloVe 转换、打分、汇总
tests/               pytest 65 项，不需要 GPU，约 6 秒
docs/                原理笔记与踩坑记录
submissions/         16 份 Kaggle 提交文件，按方法分目录
results/             分数、逐 epoch 指标、对比表
kaggle_tutorial/     Kaggle 入门教程的 Python 3 复现
notebooks/           可直接导入 Kaggle 的 Notebook
```

需要自己准备的四个目录都在 `.gitignore` 里：`corpus/imdb/`、`glove/`、`pickle/`、`models/`。

## 安装

```bash
git clone https://github.com/tguo0326/GYT-NLP-developing.git
cd GYT-NLP-developing
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` 里的 torch 是 CPU 版，要用 CUDA 按官网索引重装：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

没有 GPU 也能跑，LSTM 系慢 20~50 倍，参数高效微调那四个基本跑不动。
GloVe 准备阶段磁盘峰值约 10 GB，转换完删掉 zip 和 txt 后长期占用 2.5 GB。

## 数据

### IMDB 到 `corpus/imdb/`

要 `labeledTrainData.tsv`、`testData.tsv`、`unlabeledTrainData.tsv` 三份。

用 Kaggle 官方文件（要提交排行榜就用这个）：

```bash
# 先在 https://www.kaggle.com/competitions/word2vec-nlp-tutorial 点 Join Competition
pip install kaggle        # API token 放到 ~/.kaggle/kaggle.json
kaggle competitions download -c word2vec-nlp-tutorial -p corpus/imdb
cd corpus/imdb && unzip '*.zip' && cd ../..
```

或者从公开的 Stanford aclImdb 重建，不需要 Kaggle 账号：

```bash
curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar xzf aclImdb_v1.tar.gz
python tools/make_local_dataset.py --imdb-dir aclImdb --output-dir corpus/imdb
```

竞赛数据就是 aclImdb 的原始切分，正文逐条比对 99.9% 命中。区别在于重建出的 `id`
沿用 aclImdb 的文件名（形如 `0_2`），和官方的 `"12311_10"` 无法互推，行顺序也不同。
本仓库用的是混合配置：`testData.tsv` 用官方文件（提交需要它的 id），两份训练文件用重建版。
训练文件行顺序不同会改变 8:2 划分的具体切法，而所有模型必须落在同一套划分上才可比。

```bash
python tools/check_imdb_data.py     # 行数、字段、空值、标签分布
```

### GloVe 到 `glove/`

```bash
wget -c -O glove/glove.840B.300d.zip https://nlp.stanford.edu/data/glove.840B.300d.zip
python tools/prepare_glove.py       # 解压、转 Gensim 原生格式、自检
```

不能直接用 `KeyedVectors.load_word2vec_format(txt, no_header=True)`。840B 这份里有些
词本身带空格，Gensim 按空格切分后要求恰好 301 段，会抛 `ValueError`。脚本自己解析：
右边 300 个数当向量，剩下整段当词，2,196,017 行 0 异常。

自检里有个数字值得留意：`good ~ bad = 0.7355`。反义词的余弦相似度很高，因为 GloVe 学的是
分布相似，`good` 和 `bad` 出现在几乎相同的上下文里。所有静态词向量都有这个特点，
也是情感分类不能只看词向量余弦的原因。

## 运行

```bash
python experiments/preprocess.py                 # 产出 pickle/imdb_glove.pickle3
python experiments/baseline.py                   # BoW 与 TF-IDF

python experiments/glove/cnn.py
python experiments/glove/lstm.py
python experiments/glove/gru.py
python experiments/glove/cnnlstm.py
python experiments/glove/attention_lstm.py       # --show-attention 导出注意力权重
python experiments/glove/transformer.py
python experiments/glove/capsule_lstm.py

python experiments/finetune/distilbert.py        # 三个里最快，先跑这个
python experiments/finetune/bert.py
python experiments/finetune/roberta.py

python experiments/peft/lora.py --probe-steps 20 # 先探显存，几十秒
python experiments/peft/lora.py                  # 约 55 分钟
python experiments/peft/adalora.py --lr 5e-4
python experiments/peft/ptuning.py
python experiments/peft/prefix.py --epochs 6 --lr 3e-4

python tools/score_submissions.py --model all    # 打分，写 results/test_scores.csv
python tools/collect_results.py                  # 汇总对比表并同步 README
python -m pytest tests/ -q
```

前 12 个模型在 T4 上合计约 75 分钟（不含 GloVe 下载转换的 25 分钟），
参数高效微调那四个合计约 3 小时。

每个脚本跑完留下 `logs/<name>.log`、`results/<name>_history.csv`、
`results/<name>_summary.json`、`results/<name>_submission.csv`，
以及权重（GloVe 系是 `models/<name>_best.pt`，PEFT 是 `models/<name>_peft/` 里的 adapter）。

显存不够就调小 `--batch-size`，梯度累积步数会自动补到等效 32，和其他模型保持同一口径。
上限由 `core/mem_guard.py` 硬限在 13.5 GB，越线主动中止，被中止的运行不写结果，
不会混进对比表。

对一条新评论做预测：

```bash
python experiments/glove/cnn.py --predict "This film was a masterpiece from start to finish."
python experiments/finetune/roberta.py --predict "Boring, predictable and far too long."
```

## 踩过的坑

完整 34 条在 [docs/troubleshooting.md](docs/troubleshooting.md)。最值得记的是
「不报错但结果是错的」这一类，占了一半以上。

| 现象 | 原因 |
|---|---|
| 提交文件格式全对，分数等于随机 | `group_by_length=True` 也作用于 `predict`，概率按长度重排后与 id 逐行错位 |
| 重载 adapter 后 AUC 0.31，反相关 | DeBERTa 的 pooler 是随机初始化的，peft 只保存 classifier，重载时又换了一个随机 pooler |
| AdaLoRA 准确率 0.5094 | 默认挂载点比 LoRA 宽 6 倍；`update_and_allocate()` 要每步手动调；学习率得比 LoRA 高 5 倍 |
| 填到 512 后 `states[-1]` 取到的全是 PAD | 句尾信息被三百多个 PAD 冲掉，要用 `pack_padded_sequence` |
| CNN-LSTM 在无序集合上做顺序建模 | `permute([1,0,2])` 把卷积核维度当成了时间步 |
| Attention 权重随 batch_size 变化 | `softmax(dim=1)` 归一化到了 batch 维，应该沿时间轴 |
| Transformer 的词表退化成 26 个字母 | 分词函数返回字符串而不是词列表，`Vocab.build` 逐字符建表 |
| 验证准确率虚低且每次不同 | 少了 `model.eval()`，验证时 Dropout 还在丢神经元 |
| 显存单调增长直到 OOM | `train_loss += loss` 累加的是带计算图的张量 |
| 官方文件和 aclImdb 只有 53% 命中 | 官方把正文引号转义成 `\"`，还原后是 99.9% |

这些都有回归测试钉住。评测口径另外三点：OOV 率按词型算 22.23%、按词次算 0.43%，
报数必须说清口径；`read_csv` 要显式 `quoting=csv.QUOTE_NONE`，影评里全是引号，
按默认规则解析会行数正常但内容错位；划分必须固定 `random_state` 并分层，否则跨模型不可比。

## 文档

- [docs/peft-lora.md](docs/peft-lora.md) — LoRA 原理、显存账、四种方法对比、踩坑
- [docs/troubleshooting.md](docs/troubleshooting.md) — 34 条问题与解决
- [docs/from-bow-to-llm.md](docs/from-bow-to-llm.md) — One-Hot 到上下文词向量的脉络
- [docs/glove-word2vec-bow.md](docs/glove-word2vec-bow.md) — 三种文本表示的区别
- [docs/learning-summary.md](docs/learning-summary.md) — 学习总结
- [docs/results.md](docs/results.md) — Kaggle 教程复现的完整实测
- [docs/kaggle-gpu.md](docs/kaggle-gpu.md) — Kaggle 免费算力
- [results/comparison.md](results/comparison.md) — 自动生成的对比表

## 参考

- Kaggle: [Bag of Words Meets Bags of Popcorn](https://www.kaggle.com/competitions/word2vec-nlp-tutorial)
- Maas et al., *Learning Word Vectors for Sentiment Analysis*, ACL 2011
- Pennington et al., *GloVe: Global Vectors for Word Representation*, EMNLP 2014
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021
- Zhang et al., *AdaLoRA: Adaptive Budget Allocation for PEFT*, 2023
- Li & Liang, *Prefix-Tuning: Optimizing Continuous Prompts for Generation*, 2021
- Liu et al., *GPT Understands, Too*, 2021
- Chen et al., *Training Deep Nets with Sublinear Memory Cost*, 2016
