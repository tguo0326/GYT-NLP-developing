# 未纳入任务清单的原始脚本

压缩包里这三个脚本不在任务清单（任务 11 只列了 `imdb_transformer.py`、
`imdb_bert_trainer.py`、`imdb_distilbert_trainer.py`、`imdb_roberta_trainer.py`、
`imdb_capsule_lstm.py`），所以**保持原样、未做修复**，放在这里备查：

| 文件 | 内容 | 已知问题 |
|---|---|---|
| ` imdb_bert_native.py` | 不用 HF Trainer，手写 BERT 微调循环 | 写死 `.cuda()`；旧版 transformers API |
| `imdb_bert_scratch.py` | 从零实现 BERT 的部分组件 | 同上，且未接通 IMDB 数据 |
| `imdb_distilbert_native.py` | 手写 DistilBERT 微调循环 | 同上 |

要跑 BERT 系列请用根目录下已修复的 `imdb_bert_trainer.py` /
`imdb_distilbert_trainer.py` / `imdb_roberta_trainer.py`，它们共用 `hf_trainer.py`。
