# R-Drop / SCL + unsloth 封装的 LoRA

对应老师第二次任务：**unsloth 封装的 LoRA + Regularized Dropout (R-Drop) + Supervised Contrastive Learning (SCL)**，
并且要求"自己继承 huggingface 的类，修改 forward 或 compute_loss，之后用 unsloth 调用自己封装的，再挂 lora"。

## 任务对照

| 老师的要求 | 落在哪 |
|---|---|
| 试试 unsloth 来封装的 lora | `imdb_unsloth_lora.py` 的 `FastModel.from_pretrained` + `FastModel.get_peft_model`；unsloth 装在仓库外的独立 venv，四组实验都用它跑过一遍 |
| Regularized Dropout | `losses.rdrop_kl_loss`；两条实现路线：`imdb_bert_rdrop.py`（改 forward）、`trainers.RegularizedTrainer`（改 compute_loss） |
| Supervised Contrastive Learning | `losses.SupConLoss` / `SCLLoss`；同样两条路线：`imdb_bert_scl.py`、`RegularizedTrainer` |
| 自己继承 huggingface 的类，改 forward 或 compute_loss | 改 `forward` → `BertForRDrop` / `BertForSCL`；改 `compute_loss` → `RegularizedTrainer` |
| 用 unsloth 调用自己封装的，再挂 lora | `imdb_unsloth_lora.py`：unsloth 加载骨干 → 挂 LoRA → 用自己的 `RegularizedTrainer` 算 loss |
| 结果记录 Kaggle submission test acc（上一轮） | `submissions/17_rdrop_scl/` 共 10 份；`score_local.py` 先用本地带标签的测试集打了分 |

## 文件结构

| 文件 | 作用 |
|---|---|
| `losses.py` | `SupConLoss`（官方实现，修了 nan / 数值问题）、`SCLLoss`（做 L2 归一化 + view 补齐的封装）、`rdrop_kl_loss`（对称 KL） |
| `data.py` | IMDB tsv 读取、train/val 分层切分、tokenize、Kaggle submission 写盘 |
| `utils.py` | logging、随机种子、accuracy 指标 |
| `imdb_bert_rdrop.py` | **路线①**：继承 `BertPreTrainedModel`，在 `forward` 里做 R-Drop |
| `imdb_bert_scl.py` | **路线①**：继承 `BertPreTrainedModel`，在 `forward` 里做 SCL |
| `trainers.py` | **路线②**：继承 `Trainer`，重写 `compute_loss`，可选 `none/rdrop/scl/both`；另含绕开 unsloth monkey-patch 的安全 `prediction_step` |
| `imdb_unsloth_lora.py` | **最终目标**：unsloth `FastModel` 加载骨干 → `get_peft_model` 挂 LoRA → `RegularizedTrainer` |
| `run_all.sh` | 路线②四组对比实验的复现脚本（除 `--reg` 外超参一致） |
| `run_route1.sh` | 路线①三组对比实验的复现脚本（BERT-base 全量微调） |
| `score_local.py` | 用本地 `testDataWithLabels.tsv` 给 `submissions/17_rdrop_scl/*.csv` 打分，按实验组分表输出 |

两条路线的区别：路线①要为每种骨干网络重写一次 `forward`；路线②不碰模型内部，
任何 `AutoModelForSequenceClassification` 套 LoRA / unsloth 之后都能直接用，所以最终脚本走路线②。

## 数据

和仓库其他阶段共用 `corpus/imdb/`（不入库，用 `tools/make_local_dataset.py`
或直接放 Kaggle 的 tsv 重建）。换位置用环境变量：

```bash
export IMDB_DIR=/path/to/imdb
```

提交文件和 metrics 落在 `submissions/17_rdrop_scl/`，中间产物落在 `models/`，两者路径都从
`__file__` 推导，所以从仓库根跑 `python experiments/reg/xxx.py` 也对。

## 跑法

```bash
# 路线①三组（BERT-base 全量微调）：baseline / +R-Drop / +SCL
experiments/reg/run_route1.sh

# 路线②四组（ModernBERT-large + LoRA），peft 后端
experiments/reg/run_all.sh

# 路线②四组，unsloth 后端（结果文件带 _unsloth 后缀，不会覆盖上面那批）
PYTHON=/path/to/.venv-unsloth/bin/python TAG_SUFFIX=_unsloth experiments/reg/run_all.sh

# 汇总所有结果，按实验组分表打印
python experiments/reg/score_local.py

# 单独跑一组
python experiments/reg/imdb_unsloth_lora.py --reg rdrop --rdrop_alpha 1.0
python experiments/reg/imdb_bert_scl.py --alpha 0.2 --temperature 0.3

# 本地 smoke test（只跑 32 条，score_local.py 会自动跳过这种结果）
python experiments/reg/imdb_unsloth_lora.py --reg both --limit 32 --epochs 1 --max_length 128
```

结果：`submissions/17_rdrop_scl/<tag>.csv` + `<tag>_metrics.json`（超参 + 验证指标）。

> 注意这批脚本沿用了 demo 的 `np.argmax`，交的是 **0/1 硬标签**，
> 和前 16 种做法交概率的口径不一样，算不出 AUC，所以没进根 README 那张表。
> 详见 `submissions/17_rdrop_scl/README.md`。

## 实验结果（2026-08-17，单卡 Tesla T4）

### 路线②：ModernBERT-large + LoRA + 自定义 compute_loss

LoRA r=16 / α=32 / dropout 0.05，lr 2e-4，`max_length 256`，`batch_size 16`，2 epochs，
fp16，seed 3407。**除 `--reg` 外超参完全一致。** 两个后端各跑一遍：

复现：`run_all.sh`（peft）、`PYTHON=<venv>/bin/python TAG_SUFFIX=_unsloth run_all.sh`（unsloth），
然后 `python experiments/reg/score_local.py`。

| 正则 | 关键超参 | unsloth val | unsloth test | peft val | peft test |
|---|---|---|---|---|---|
| —（LoRA baseline） | — | 0.9474 | 0.9537 | 0.9464 | **0.9548** |
| R-Drop | α=1.0 | **0.9484** | **0.9540** | 0.9482 | 0.9546 |
| SCL | α=0.2, τ=0.3 | 0.9480 | 0.9538 | 0.9472 | 0.9542 |
| R-Drop + SCL | 同上 | 0.9476 | 0.9541 | 0.9466 | （csv 被覆盖，见下） |

**这四组没有观察到 R-Drop / SCL 带来提升**（路线①的全量微调是另一回事，见下）。
7 个可用 test acc 全落在 0.9537~0.9548，
极差 0.11 个百分点 = 25000 条里差 28 条。两个后端上 val 都是 R-Drop 略高、test 都是 baseline 略高，
**排序在 val 和 test 之间反转，说明差异完全在单 seed 的随机波动内，不能声称任何一方更好。**

> peft 版 `both` 的 submission csv 被一次 `--limit 32` 的 smoke test 覆盖了（当时还没加
> `--tag_suffix`）。val acc 0.9466 从训练日志里还能取到，test acc 要重跑 82 分钟才能补，暂缺。

### 路线①：BERT-base 全量微调 + 改 forward

同样 `max_length 256` / `batch 16` / 2 epochs，但是**全量微调**（不挂 LoRA），
所以 lr 用 2e-5。baseline 用 `imdb_bert_scl.py --alpha 0`（SCL 权重为 0 就退化成纯交叉熵）。
复现：`experiments/reg/run_route1.sh`。

| setting | val acc | test acc | Δtest | 训练耗时 |
|---|---|---|---|---|
| baseline（纯 CE） | 0.9210 | 0.9205 | — | 11 min |
| + R-Drop α=1.0 | **0.9270** | **0.9241** | **+0.0036** | 18 min |
| + SCL α=0.2, τ=0.3 | 0.9238 | 0.9210 | +0.0006 | 11 min |

**这组里 R-Drop 是有效的：val +0.60 个点、test +0.36 个点（25000 条里多对 90 条），
而且 val 和 test 的排序一致**，比 LoRA 那组 0.1 个点的抖动明确得多。

原因和我前面的判断一致：BERT-base 全量微调要更新 1.1 亿个参数，比只训 0.4% 参数的 LoRA
容易过拟合得多，抗过拟合的正则才有发挥空间。**也就是说 R-Drop / SCL 的收益和"模型有多容易
过拟合"直接相关，LoRA 本身已经是一种很强的正则（参数量被限制住了），再叠正则收益就很小。**
这是把两条路线的结果放在一起才看出来的，单看任何一条都得不出这个结论。

SCL 在这组里只有 +0.06 个点，基本还是噪声。可能的原因：α=0.2 和 τ=0.3 都是照示例拍的没调过；
二分类下正样本对太容易凑（batch 16 里平均 8 个同类），对比任务本身太简单，学不到什么。

### 两个后端的实测差异

| | peft | unsloth |
|---|---|---|
| 显存 | 2.7 G | **1.7 G**（−37%） |
| 速度 | 0.85 s/it | 1.12 s/it（慢 32%） |
| 可训练参数 | 0.4% | 1.79%（把分类头也设为可训练） |
| 训练 loss 日志 | 每 20 步一条 | **被 patch 掉**，画不了 loss 曲线 |

两个后端的 baseline val acc 只差 0.1 个点，说明 unsloth 那条 code path 的实现是对的。

### 还欠的实验（按性价比排序）

1. **多 seed**：现在每组只有 n=1。LoRA 那四组 0.1 个点的差异必须靠 3~5 个 seed 的
   mean±std 才能定性；路线① R-Drop 那 +0.36 个点也需要多 seed 确认不是运气。
2. **少样本设定**：每类只取 100 / 500 / 1000 条。数据越少越容易过拟合，按下面第 6 节
   那个"过拟合空间"的逻辑，这是最可能让 LoRA + 正则也出效果的方向，
   也是 SCL 那篇论文（arXiv 2011.01403）的主实验设定。
3. **α / τ 扫描**：R-Drop 论文在 NLU 上取 α∈[1,5]，我只试了 1.0；SCL 的 α=0.2、τ=0.3
   是照示例拍的，一次都没调。SCL 两条线上都没效果，很可能就是这个原因。
4. **补 peft `both` 的 test acc**（csv 被覆盖了，重跑 82 分钟）。
5. 长度截断从 256 提到 512（T4 上四组要 ~17h，这次没做）。

> 备注：表里 test acc 是用 `corpus/imdb/testDataWithLabels.tsv` 在本地打的分（`score_local.py`），
> 方便立刻看到结果；要提交的 csv 在 `submissions/17_rdrop_scl/`，提交后以官方分数为准。
> 注意这批交的是硬标签，算不出 AUC，所以没进根 README 那张主表。

### 已知的口径问题

peft 那批里 `scl` / `both` 的 `eval_loss`（0.76）和另两组（0.157）不可比，
因为 `RegularizedTrainer.compute_loss` 在验证阶段也加了 SCL 项。acc 不受影响。
unsloth 那批不存在这个问题 —— 那条路径下 `safe_prediction_step=True` 生效，
验证走的是纯 CE，四组 loss（0.1535~0.1588）可以直接比。
要让两边口径统一，把 SCL 那一项也加上 `model.training` 判断就行。

## 显存不够时（老师给的顺序）

1. 降 `--batch_size`
2. 提 `--grad_accum`（等效大 batch；SCL 需要真实 batch 里有同类样本，所以**不能只靠累积**，真实 batch 建议 ≥16）
3. `--load_in_4bit`
4. 降 `--max_length`

gradient checkpointing 默认已开（unsloth 路径用 `"unsloth"` 模式，peft 路径用 `gradient_checkpointing_enable()`）。

## 需要注意的几个点（对照老师给的示例代码，这里都改掉了）

1. **R-Drop 必须有 dropout**。unsloth 例子里 `lora_dropout=0`，这样两次前向输出完全一样、KL 恒为 0，
   R-Drop 直接失效。本脚本默认 `--lora_dropout 0.05`，`reg=rdrop` 且 dropout=0 时会打 warning。
2. **R-Drop 的 KL 用 `batchmean` 而不是 `sum`**，否则 loss 量级随 batch size 漂移，α 没法调。
   总损失按论文写成 `0.5*(CE1+CE2) + α*KL`。
3. **评测时不做第二次前向**：`model.eval()` 下 dropout 已关，跑两次没意义。
4. **SCL 的输入要 L2 归一化**，且 `SupConLoss` 需要 `[bsz, n_views, dim]`。示例里直接把 `[bsz, dim]`
   的 `pooled_output` 未归一化传进去，除以 τ=0.07 后 logits 会爆。
5. **`SupConLoss` 的 nan**：某一类在 batch 里只出现一次时 `mask.sum(1)==0`，原始实现会除零得 nan。
   这里把无正样本的 anchor 丢掉。
6. **`datasets.load_metric` 已被删除**，统一用 `evaluate.load("accuracy")`。
7. **标签列名必须是 `labels`**（HF 约定）；`compute_loss` 里自己 pop 了 labels，所以
   `TrainingArguments(label_names=["labels"])` 要显式声明，否则 eval 阶段拿不到标签。
8. **ModernBERT / DeBERTa 没有 BERT 的 pooler**，不能 `outputs[1]` 取句向量，
   统一用 `masked_mean_pool` 对 `last_hidden_state` 做 mask 平均。
9. **LoRA 的学习率要比全量微调高一个量级**（2e-4 vs 2e-5），只训低秩增量。
10. `--limit` 只用于 smoke test，正式实验必须去掉（示例里写死的 `train[0:20]` 是调试残留）。

## 环境备注

- 仓库主环境（conda，Python 3.13）：`transformers 4.51.3` / `torch 2.8.0+cu128` / `peft 0.20.0`，没装 unsloth。
- unsloth 装在独立 venv（不在仓库里，我放在 `科研专用/LoRA/.venv-unsloth`），
  Python 3.10 + `torch 2.11.0+cu130` +
  `transformers 5.5.0` + `unsloth 2026.8.18`。单独建 venv 是因为 unsloth 会把 torch 和
  transformers 的版本钉死，而 `科研专用/` 下好几个项目共用那个 conda 环境，不能动它。
- `imdb_unsloth_lora.py` 检测不到 unsloth 就自动退回 `transformers + peft`，其余代码路径不变，
  所以同一个脚本在两个环境里都能跑。
- 跑 unsloth 要带两个环境变量（`run_all.sh` 里已经设了）：

  ```bash
  export CC=/usr/bin/gcc                        # 否则 Triton 会挑 conda 的交叉编译器
  export CPATH=/usr/include/x86_64-linux-gnu    # 那个编译器看不到系统 Python 头文件
  ```

  不设的话 Triton 编译 `cuda_utils.c` 失败，报 `pyconfig.h: No such file or directory`，
  然后 fallback 到 CPU，unsloth 的 kernel 全部失效（但不会报错，很容易被忽略）。
- 主环境的 Keras 3 和 transformers 的 TF 后端冲突，脚本里 `USE_TF=0` 规避。
- 别在脚本里硬编码 `HF_ENDPOINT=https://hf-mirror.com`。这个镜像对一部分文件不返回 etag，
  `huggingface_hub` 会当成"连不上网"直接拒收（`LocalEntryNotFoundError`），而直连
  huggingface.co 在这台机上是通的。要用镜像就在 shell 里临时设。

## 要读的论文

- LoRA: https://arxiv.org/abs/2106.09685
- R-Drop: https://arxiv.org/abs/2106.14448
- SupCon（视觉，损失原型）: https://arxiv.org/abs/2004.11362
- SCL for PLM fine-tuning（NLP 用法、λ 与 τ 取值）: https://arxiv.org/abs/2011.01403

---

# 原理笔记与本次改动记录

这一节是我读完论文以后自己整理的，一是把 SCL、R-Drop、LoRA、unsloth 这几件事的原理串起来，
二是把这次为什么这么改、改在哪、怎么用记下来，免得过一段时间自己都想不起来当初为什么这么写。

## 1. 为什么要挂 LoRA

全量微调 ModernBERT-large 要更新全部 3.95 亿个参数，光 AdamW 的一阶二阶动量就是参数量的两倍，
一张 15G 的 T4 根本放不下。LoRA 的想法是：微调时权重的变化量 ΔW 其实是低秩的，
没必要用一个满秩矩阵去表示它。所以把 ΔW 拆成两个瘦矩阵的乘积：

```
W' = W + BA        A: r×k    B: d×r    r << min(d, k)
```

原来的 W 冻住不动，只训 A 和 B。r=16 时可训练参数掉到 0.4%（unsloth 那边 1.79%，
因为它把分类头也一起训了），优化器状态也跟着缩小同样的比例。推理时可以把 BA 加回 W，
不增加任何延迟，这点比 adapter 和 prefix-tuning 干净。

A 用高斯初始化、B 初始化成 0，所以训练刚开始 BA=0，模型行为跟原始预训练权重完全一致，
不会一上来就把预训练知识打乱。`lora_alpha` 是缩放系数，实际用的是 `alpha/r` 这个比值，
所以调 r 的时候一般让 alpha 跟着按比例变（我这里 r=16 / alpha=32，比值 2）。

LoRA 只训低秩增量，梯度信号弱，学习率要比全量微调高一个量级：**2e-4 而不是 2e-5**。
这个很容易照全量微调的习惯写错，写成 2e-5 的话 loss 掉得非常慢。
这次两条路线的 lr 就是按这个来的：路线② LoRA 用 2e-4，路线① 全量微调用 2e-5。

## 2. 为什么换 unsloth

peft 已经能挂 LoRA 了，换 unsloth 的动机是显存和速度。它主要做三件事：
把 attention、MLP、cross entropy 这些算子用 Triton 手写成融合 kernel，减少中间张量的读写；
自己实现 gradient checkpointing（`use_gradient_checkpointing="unsloth"`），
反向传播时不保存中间激活而是现场重算，用算力换显存；再加上对 4bit 量化加载的支持。

用法就三步，和 peft 的写法几乎一一对应：

```python
model, tokenizer = FastModel.from_pretrained(     # 1. 替代 AutoModel.from_pretrained
    model_name="answerdotai/ModernBERT-large",
    auto_model=AutoModelForSequenceClassification,  # 分类任务要显式指定
    num_labels=2, max_seq_length=256, dtype=torch.float16,
)
model = FastModel.get_peft_model(                # 2. 替代 peft.get_peft_model
    model, r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["Wqkv", "Wo", "Wi"],         # ModernBERT 的线性层名字
    use_gradient_checkpointing="unsloth",
    task_type="SEQ_CLS",
)
trainer = RegularizedTrainer(model=model, ...)   # 3. Trainer 照常用
```

**实测结论：显存从 2.7G 降到 1.7G（省 37%），但速度反而慢了 32%。** 原因在日志里写得很清楚：

```
Dropout = 0 is supported for fast patching. You are using dropout = 0.05.
Unsloth will patch all other layers, except LoRA matrices, causing a performance hit.
```

unsloth 的融合 kernel 只在 `lora_dropout=0` 时才走快速路径。这就和 R-Drop 撞了 ——
**R-Drop 必须有 dropout 才有意义**，两个需求是直接矛盾的。老师给的示例里写
`lora_dropout=0`（照抄 unsloth 官方例子），如果直接拿那份配置去做 R-Drop，
两次前向的输出会完全一样、KL 恒等于 0，等于什么都没做。所以我这边默认给 0.05，
并且在 `reg=rdrop` 且 dropout=0 时打一条 warning。

另外两个坑：unsloth 会 patch 掉 Trainer 的日志回调，训练 loss 和中间 epoch 的 eval 指标
一条都不打，画 loss 曲线得回去用 peft 那版的日志；它还 patch 了模型的 `prediction_step`，
所以 `RegularizedTrainer` 里留了个 `safe_prediction_step`，直接调 `model(**inputs)`
自己解包 logits，绕过它。

对这个任务来说 unsloth 不算划算（encoder 模型层数浅、序列短，它的优化主要是冲着
decoder LLM 去的）。真正需要它的场景是显存卡死、模型放不进去的时候。

## 3. R-Drop 的原理

dropout 在训练时随机丢神经元、推理时不丢，这本身就造成了训练和推理的不一致：
训练看到的是一堆"子网络"，推理用的是完整网络。R-Drop 的做法是把这个不一致直接写进损失函数。

同一批输入过两次网络。因为两次的 dropout mask 是独立采样的，会得到两个不同的预测分布
p1 和 p2。除了各自算交叉熵，再加一项对称 KL 把它们拉近：

```
L = 0.5 * (CE(p1, y) + CE(p2, y)) + α * 0.5 * (KL(p1‖p2) + KL(p2‖p1))
```

直观理解：强迫模型对"丢掉哪些神经元"这件事不敏感。任意两个子网络的输出都得一致，
等价于给参数加了一个隐式的正则约束，比单纯用 dropout 更强。用对称 KL 而不是单向的，
是因为 p1 和 p2 地位对等，没有理由让谁去拟合谁。

实现时踩到的三个点：

- **KL 的 reduction 要用 `batchmean`，不能用 `sum`。** 用 sum 的话 loss 量级会随 batch size
  线性漂移，α 就没法调了（换个 batch size 之前调好的 α 全废）。
- **算 KL 走 log 空间**：`F.kl_div(log_p, log_q, log_target=True)`，比先 softmax 再取 log
  数值稳定。老师示例里 `F.softmax(target, dtype=...)` 还漏了 `dim=-1`，那样会按最后一维之外
  的默认维度归一化，结果是错的。
- **验证的时候不要跑第二次前向**。`model.eval()` 下 dropout 已经关了，两次前向输出完全相同，
  KL 恒为 0，白算一遍。代码里用 `self.training` / `model.training` 卡住。

代价是训练时间翻倍（实测 1.12 s/it → 2.21 s/it，正好 2 倍），因为每步要前向两次。

## 4. SCL 的原理

交叉熵只关心"分类边界画对没有"，不关心特征空间长什么样。一个模型可以把两类分对，
但同类样本的表示在空间里散得很开、不同类的挤在一起，这种表示的泛化性和鲁棒性都差。
SCL 就是额外要求：**batch 里同标签的句向量互相靠近，不同标签的互相推远。**

先看无监督的 SimCLR：一个样本做两次数据增强得到两个 view，互为正样本，
batch 里其他所有样本都是负样本。SCL 把"正样本"的定义从"同一个样本的另一个增强"
换成"同一个类别的所有样本"，这样有标签的信息就用上了，而且正样本不再只有一个：

```
L_SCL = Σ_i  (-1/|P(i)|) Σ_{p∈P(i)}  log [ exp(z_i·z_p/τ) / Σ_{a≠i} exp(z_i·z_a/τ) ]
```

- `z` 是 L2 归一化后的句向量，所以 `z_i·z_p` 就是余弦相似度，范围 [-1, 1]
- `P(i)` 是 batch 里和 i 同标签的样本集合（不含 i 自己）
- 分母跑遍 batch 里除自己以外的全部样本，正负都算进去
- 外面那个 `1/|P(i)|` 是对所有正样本取平均，这是 SupCon 论文里说的 `L_in` 之外那个
  数值上更好的形式（把求和放在 log 外面）

**τ（temperature）是这里最关键的超参。** 它在指数里做除法，作用是控制分布的尖锐程度：
τ 越小，softmax 越尖，模型会把注意力集中在最难的那些负样本上（hard negative mining 的效果），
但太小会导致梯度不稳、对噪声标签敏感；τ 越大分布越平，所有负样本被平等对待，
学出来的表示区分度不够。视觉侧常用 0.07，文本侧那篇论文用 0.1~0.5，我这里取 0.3。

代码里 `logits - logits_max.detach()` 那一步是纯数值技巧，不改变数学结果：
softmax 对输入整体平移不变，减掉每行最大值可以避免 `exp()` 溢出。`detach()` 是因为
这个最大值只是个常数偏移，不该有梯度流过去。

总损失是交叉熵和 SCL 的加权和：

```
L = CE + α * L_SCL
```

（论文里写成 `(1-λ)·CE + λ·L_SCL`，λ 取 0.9，和这里的 α 只差一个参数化方式。我沿用了
老师示例里 `CE + α·SCL` 的写法，α=0.2。）

实现时踩到的四个点：

- **特征必须先 L2 归一化。** `SupConLoss` 假定输入落在单位球面上，直接把没归一化的
  `pooled_output` 传进去，`z_i·z_p` 会是几十上百的量级，再除以 τ=0.07，`exp()` 直接溢出。
  老师示例里就是没归一化，这个必须补上。
- **形状要求是 `[bsz, n_views, dim]`**，三维。示例里传的是二维的 `[bsz, dim]`，会直接报错。
  文本这边没有做两个增强 view，所以 `unsqueeze(1)` 补一个 n_views=1 的维度就行。
- **原版实现有除零得 nan 的风险。** 如果某个类在这个 batch 里只出现一次，那它没有正样本，
  `mask.sum(1)` 是 0，`(mask*log_prob).sum(1) / mask.sum(1)` 就是 nan，整个训练直接崩。
  我在 `losses.py` 里把无正样本的 anchor 过滤掉了。二分类 + batch 16 基本不会触发，
  但类别多或者 batch 小的时候一定会踩。
- **batch size 不能太小。** SCL 完全依赖 batch 内的同类样本构造正样本对，
  batch=2 的话大概率一个正样本都凑不出来，这一项等于没加。而且 `gradient_accumulation_steps`
  在这里救不了 —— 累积只是把梯度攒起来，每次前向的 batch 还是那么小，
  对比是在单次前向的 batch 内做的。所以真实 batch 建议至少 16，显存不够就降序列长度，
  别降 batch。

取句向量的方式也要注意：BERT 有 pooler 层，`outputs[1]` 就是 `[CLS]` 过 tanh 之后的结果；
但 ModernBERT 和 DeBERTa 没有 pooler，`outputs[1]` 拿到的不是句向量。
所以路线②里统一用 `masked_mean_pool` 对 `last_hidden_state` 做 mask 平均池化，
把 padding 位置排除掉，这样换骨干不用改代码。

## 5. 三个东西怎么拼在一起

改损失函数有两条路：改模型的 `forward`，或者改 Trainer 的 `compute_loss`。
两条我都写了（`imdb_bert_*.py` 是前者，`trainers.py` 是后者），最后 unsloth 那套用的是后者。
理由：改 `forward` 要为每种骨干各写一个类（BERT 有 pooler、ModernBERT 没有、
参数名也不一样），而且模型被 peft 和 unsloth 包了好几层之后再去改它内部很别扭；
改 `compute_loss` 完全不碰模型，任何 `AutoModelForSequenceClassification` 套上 LoRA、
套上 unsloth 之后都能直接用。

`RegularizedTrainer.compute_loss` 的流程：

```python
labels = inputs.pop("labels")          # 自己接管 labels，不让模型内部算 loss，否则重复计算
outputs = model(**inputs)
loss = CE(outputs.logits, labels)

if use_scl:
    features = masked_mean_pool(hook 抓到的 last_hidden_state, attention_mask)
    loss += scl_alpha * SCLLoss(features, labels)

if use_rdrop and model.training:
    logits2 = model(**inputs).logits    # 第二次前向，dropout 重新采样
    loss = 0.5 * (loss + CE(logits2, labels)) + rdrop_alpha * symmetric_KL(logits, logits2)
```

SCL 需要最后一层的隐藏状态。最直觉的写法是 `model(**inputs, output_hidden_states=True)`，
但这样它会把全部 28 层的输出都留在显存里，正好把 gradient checkpointing 省下来的显存
又吃回去 —— batch 16 / len 256 直接 OOM。改成在骨干上挂一个 forward hook 只抓最后一层，
显存立刻回到和 baseline 一样的水平（1.7G）。找骨干模块用 HF 的 `base_model_prefix` 约定
（ModernBERT 是 `model`、DeBERTa 是 `deberta`、BERT 是 `bert`），所以 `find_encoder()`
对这几种模型都通用。

另外因为 `compute_loss` 里把 labels pop 掉了，`TrainingArguments` 必须显式声明
`label_names=["labels"]`，否则 Trainer 在评测阶段找不到标签，`compute_metrics` 拿到的
references 是空的。

## 6. 结果怎么解读

跑完两条路线一共 7 组，结论比单看一条线清楚得多：

| | 可训练参数 | R-Drop 的 Δtest |
|---|---|---|
| ModernBERT-large + LoRA | 0.4% | +0.03 个点（噪声） |
| BERT-base 全量微调 | 100% | **+0.36 个点** |

**R-Drop 在全量微调下有效，在 LoRA 下测不出来。** 我理解这不是巧合：R-Drop 和 SCL 本质上都是
抗过拟合的手段，而 LoRA 本身就是一种很强的正则 —— 它把可更新的自由度压到了 0.4%，
模型压根没什么过拟合的余地，再叠一层正则自然没什么可省的。全量微调要更新 1.1 亿个参数，
过拟合空间大，正则才有用武之地。

LoRA 那四组 test acc 挤在 0.9537~0.9548（极差 0.11 个百分点），而且 val 上的排序和 test 上的
排序还是反的，只能当作噪声看。另一个因素是任务太简单：IMDB 情感二分类对 ModernBERT-large 来说
94~95% 已经接近 256 token 截断下的天花板，剩下的错误多半是长评论被截断或者标签本身有噪声，
正则化改善不了这些。

SCL 在两条线上都只有 0.0~0.06 个点，暂时看不出效果。我怀疑主要是二分类的锅：batch 16 里
平均有 8 个同类样本，正样本对随手就能凑出来，对比这个任务本身太简单，学不到额外的东西。
多分类（比如 20newsgroups）或者少样本设定下应该更能体现它的价值。

要真正验证这两个方法，接下来该做的（按性价比排）：

1. **多 seed。** 现在每组只有 n=1，0.1 个点的差异毫无意义。每组 3~5 个 seed 报 mean±std，
   这是最要紧的一步。
2. **少样本设定。** 每类只取 100 / 500 / 1000 条。SCL 那篇论文的主实验就是少样本，
   R-Drop 也是数据越少收益越明显。数据越少越容易过拟合，按上面那个"过拟合空间"的逻辑，
   这也是最可能让 LoRA + 正则也出效果的方向。
3. **扫 α 和 τ。** R-Drop 论文在 NLU 任务上 α 取 1~5，我只试了 1.0；SCL 的 α=0.2、τ=0.3
   也是照示例拍的，没调过。
4. 序列长度提到 512。这次受 T4 算力限制没做（512 下四组要跑 17 小时）。
