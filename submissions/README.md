# Kaggle 提交文件

竞赛：https://www.kaggle.com/competitions/word2vec-nlp-tutorial
指标是 ROC-AUC，所以每份 `submission.csv` 交的是正面情感概率，不是 0/1 标签。

每个方法一个子目录，里面是 `submission.csv` + `summary.json` + 说明。
下面按 AUC 排序，分数是本地用 aclImdb 公开标签算的（24,961 / 25,000 条能对齐），与排行榜同源。

| 目录 | 方法 | 阶段 | 测试集准确率 | ROC-AUC | 可训练参数 |
|---|---|---|--:|--:|--:|
| `16_lora/` | DeBERTa-v3-large + LoRA | 阶段三 | 0.9633 | 0.9924 | 2,624,514 |
| `18_scl_moco/` | LoRA + 监督对比 / MoCo，真实 batch=4 | 阶段四 | 0.9635 | 0.9924 | 3,805,314 |
| `14_p_tuning/` | DeBERTa-v3-large + P-Tuning | 阶段三 | 0.9537 | 0.9906 | 302,338 |
| `15_adalora/` | DeBERTa-v3-large + AdaLoRA | 阶段三 | 0.9607 | 0.9904 | 4,198,914 |
| `12_roberta/` | RoBERTa 全量微调 | 阶段二 | 0.9369 | 0.9834 | 124,647,170 |
| `11_bert/` | BERT 全量微调 | 阶段二 | 0.9174 | 0.9746 | 109,483,778 |
| `10_distilbert/` | DistilBERT 全量微调 | 阶段二 | 0.9094 | 0.9703 | 66,955,010 |
| `07_attention_lstm/` | GloVe + Attention-LSTM | 阶段二 | 0.9027 | 0.965 | 902,402 |
| `09_capsule_lstm/` | GloVe + Capsule-LSTM | 阶段二 | 0.9036 | 0.9648 | 868,610 |
| `06_cnn_lstm/` | GloVe + CNN-LSTM | 阶段二 | 0.8998 | 0.9637 | 314,498 |
| `05_gru/` | GloVe + BiGRU | 阶段二 | 0.8985 | 0.963 | 565,442 |
| `04_lstm/` | GloVe + BiLSTM | 阶段二 | 0.8975 | 0.9629 | 753,602 |
| `13_prefix_tuning/` | RoBERTa-base + Prefix-Tuning | 阶段三 | 0.8964 | 0.9608 | 960,770 |
| `03_cnn/` | GloVe + TextCNN | 阶段二 | 0.8858 | 0.9579 | 461,954 |
| `08_transformer/` | GloVe + Transformer | 阶段二 | 0.8815 | 0.9541 | 1,342,026 |
| `01_bag_of_words/` | Bag of Words + 随机森林 | 阶段一 |  |  | 5,000 |
| `02_tfidf/` | TF-IDF + 逻辑回归 | 阶段一 |  |  | 200,000 |

**最好的是 `16_lora/`**（LoRA，AUC 0.9924）。只交一份就交它。

`17_rdrop_scl/` 与 `18_scl_moco/` 都是一个目录装多份运行：前者是 R-Drop / SCL 那批
（交的是 0/1 硬标签，算不出 AUC，所以没进表），后者是小 batch 下的监督对比与 MoCo 四份
（纯交叉熵 / 监督对比 / 队列与动量编码器两种动量系数）。`18_scl_moco/` 那行取的是四份里
准确率最高的监督对比与 AUC 最高的 m=0.99，四份之间实际分不出差别。
