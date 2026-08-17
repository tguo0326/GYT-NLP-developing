# 17 · R-Drop / SCL + unsloth-LoRA

代码在 [`experiments/reg/`](../../experiments/reg/)，原理和改动记录写在
[`experiments/reg/README.md`](../../experiments/reg/README.md)。

## 两件必须先说清楚的事

**1. 这批交的是 0/1 硬标签，不是概率。** 前 16 种做法交的都是
`softmax` 出来的正类概率（竞赛指标是 ROC-AUC，硬标签会白扔很多分，
理由见 `tools/score_test.py`）。这一批的脚本是从老师给的 demo 改的，
沿用了 demo 的 `np.argmax`，所以只有硬标签。**因此这些结果算不出 AUC，
没有进根 README 那张对比表。** 要补 AUC 得把 `argmax` 换成 `softmax(...)[:, 1]`
再重训一遍（模型没存，`save_strategy="no"`）。

**2. 口径和前 16 种不一致，不能横向比。** 这批是 ModernBERT-large / BERT-base、
`max_length 256`、真实 batch 16 不做累积；主表是 DeBERTa-v3-large、
`max_length 384`、等效 batch 32。这一批的目的是 **同一底座下比"加不加正则"**，
组内四个设定的超参完全一致，所以组内可比，跨表不可比。

## 文件

| 前缀 | 是什么 |
| --- | --- |
| `ModernBERT-large_lora_{none,rdrop,scl,both}_unsloth.csv` | unsloth 后端，路线②（改 `compute_loss`），4 组 |
| `ModernBERT-large_lora_{none,rdrop,scl}_peft.csv` | peft 后端，同样 4 组里的 3 组（`both` 的 csv 被一次冒烟测试覆盖了） |
| `bert-base_full_{none,rdrop,scl}.csv` | 路线①（改 `forward`），BERT-base 全量微调，3 组 |
| `*_metrics.json` | 每次运行的完整超参 + 验证指标 |

`none` 是同配置的 baseline（不加正则）。路线① 的 baseline 用
`imdb_bert_scl.py --alpha 0` 跑的，SCL 权重为 0 就退化成纯交叉熵。

## 本地打分

`python experiments/reg/score_local.py` 会拿 `corpus/imdb/testDataWithLabels.tsv`
给这个目录下所有 csv 打分，按实验组分表输出。冒烟测试的结果（`limit` 不为空）会自动跳过。

| 实验组 | baseline | +R-Drop | +SCL | +两者 |
| --- | --: | --: | --: | --: |
| ModernBERT-large + LoRA (unsloth) | 0.9537 | 0.9540 | 0.9538 | 0.9541 |
| ModernBERT-large + LoRA (peft) | 0.9548 | 0.9546 | 0.9542 | — |
| BERT-base 全量微调（路线①） | 0.9205 | **0.9241** | 0.9210 | — |

**R-Drop 在全量微调上有效（+0.36 个点），在 LoRA 上测不出来（+0.03，噪声级）。**
LoRA 把可更新自由度压到 0.4%，本身就是很强的正则，没留下多少过拟合空间给 R-Drop 去省。
SCL 两条线上都没看出效果，α 和 τ 一次都没调过，暂时只能说"未观察到"。
