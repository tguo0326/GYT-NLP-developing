# 阶段三：参数高效微调（PEFT）—— LoRA / AdaLoRA / P-Tuning / Prefix-Tuning

> 阶段二把 BERT / RoBERTa 的**全部**参数都拿来微调，验证集准确率冲到 0.9292。
> 但这条路有个天花板：可训练参数等于模型参数，一张 15 GB 的 T4 顶多微调到
> 3 亿参数级别。真正的大模型怎么办？
>
> 这一阶段的答案是：**冻住底座，只训练插进去的一小块**。

---

## 1. 为什么全量微调撑不住

先算一笔显存账。用 Adam 优化器微调一个 `P` 参数的模型，显存至少要装：

| 项目 | 大小 | 4 亿参数模型（fp32） |
|---|---|---|
| 权重 | 4P | 1.6 GB |
| 梯度 | 4P | 1.6 GB |
| Adam 一阶动量 `m` | 4P | 1.6 GB |
| Adam 二阶动量 `v` | 4P | 1.6 GB |
| **小计（与 batch 无关）** | **16P** | **6.4 GB** |
| activation | 随 batch × 序列长度增长 | 3~8 GB |

关键在于**前四项跟 batch size 完全无关**，是硬性开销。调小 batch 只能压 activation，
压不动这 16P。所以 15.7 亿参数的 `deberta-v2-xxlarge` 全量微调需要
16 × 1.57 G ≈ **25 GB** 才刚放下权重和优化器状态，T4 根本不用试。

PEFT 直接把这笔账拆掉：底座 `requires_grad=False`，梯度和 Adam 状态**只为那
0.3% 的可训练参数分配**。16P 里的 12P（梯度 + 两份动量）几乎归零，
剩下 4P 的权重还能用 fp16 再砍一半。

这就是本项目实测到的结果——`deberta-v2-xxlarge`（15.7 亿参数）在 T4 上
**用 3.49 GB 显存跑起来了**，正是导师说的「10% 不到的显存」。

---

## 2. 四种方法，两条思路

```
                    ┌── 改权重 ──┬─ LoRA        给 W 加一个低秩增量 BA
                    │            └─ AdaLoRA     同上，但每层的秩会自适应调整
冻结底座，只训一小块 ┤
                    └── 改输入 ──┬─ P-Tuning    输入层前面拼 20 个虚拟 token
                                 └─ Prefix      每一层的 K/V 前面都拼 20 个
```

### 2.1 LoRA（Hu et al., 2021）—— 重点

论文的核心观察：微调产生的更新量 ΔW 虽然和 W 同样大，但**内在秩很低**。
既然如此，就不要去更新 W，而是把 ΔW 显式分解成两个瘦矩阵：

```
h = W·x  +  (α/r)·B·A·x

W ∈ R^{d×k}   冻结，不参与梯度
A ∈ R^{r×d}   高斯随机初始化
B ∈ R^{d×r}   全零初始化  ← 关键
r ≪ min(d,k)  本项目用 r=16
```

**为什么 B 初始化为零**：训练开始时 BA = 0，模型的输出和原底座**逐位相同**。
不会一上来就被随机噪声打乱——这也是 LoRA 训练比 Adapter 稳定的原因之一。

**参数量算一下**：DeBERTa-v3-large 的一个注意力投影是 1024×1024 = 105 万参数。
挂上 r=16 的 LoRA 后只多 2 × 1024 × 16 = **3.3 万**，是原来的 3.1%。
只给 Q 和 V 挂（论文消融实验里 Q+V 性价比最高），24 层合计 157 万参数，
占整个模型的 **0.361%**。

**α/r 这个缩放因子是干什么的**：如果只写 `B·A·x`，那么改变 r 时增量的量级也会跟着变，
学习率就得重新调。乘上 α/r 之后，r 从 8 换到 64 时增量的尺度大致不变，
超参数可以迁移。本项目 α=32, r=16，即缩放 2 倍。

**最实用的一点——推理零开销**：训完可以做

```python
merged = peft_model.merge_and_unload()   # W ← W + (α/r)·B·A
```

`BA` 被加回 `W` 里，得到一个和原底座**结构完全一样**的模型。
这是 LoRA 相比 Adapter（在层间串联额外的小网络，永久增加深度和延迟）的决定性优势：
Adapter 推理时每层都多两次矩阵乘，LoRA 合并后一次都不多。

一个底座 + N 个 adapter（每个几 MB）就能服务 N 个任务，换任务只换 adapter——
这正是导师说「后面训练模型，大部分只能挂 LoRA」的现实原因。

### 2.2 AdaLoRA（Zhang et al., 2023）

LoRA 的所有层共用同一个 r。但底层学的是通用的词法句法特征，顶层学任务特征，
需要的容量本来就不一样，平摊等于浪费。

AdaLoRA 把增量写成类 SVD 的形式，让秩本身变成可裁剪的：

```
ΔW = P · Λ · Q          Λ = diag(λ_1 ... λ_r)
```

训练中给每个三元组 `(p_i, λ_i, q_i)` 算一个**重要性分数**（梯度 × 权重的滑动平均，
近似「删掉它损失会涨多少」），定期把分数最低的置零，把预算让给更需要的层。
调度分三段：

```
0 ────── tinit ──────────────── total-tfinal ────── total
  不裁剪          每 deltaT 步裁一次           固定下来继续训
（先让重要性统计稳定）  init_r 逐步降到 target_r
```

本项目从 `init_r = 32` 起、裁到 `target_r = 16`，最终预算和 LoRA 相当——
比的就是「同样的参数预算，会不会分配」。代价是要多维护一份重要性统计，
实测比 LoRA 慢一些。

**这是四种方法里最难跑对的一个，踩了三个坑才跑出正常结果**
（第一次跑出来的验证准确率是 0.5094，等于瞎猜）。三个坑都不报错：

**坑一：默认挂载点比 LoRA 宽 6 倍，两者根本不是同一个实验。**
`target_modules` 留空时，peft 给 LoRA 的默认映射是 `{query_proj, value_proj}`，
而给 AdaLoRA 的是 `{query_proj, key_proj, value_proj, dense}`——
多了 K 投影和**所有 FFN 的 dense**：

```
LoRA     挂载 48 处   可训练  1,574,914
AdaLoRA  挂载 290 处  可训练 14,228,002   ← 9 倍
```

这样比出来的不是「同样的预算会不会分配」，而是「谁的预算大」。
所以 AdaLoRA 必须**显式写死** `target_modules=["query_proj", "value_proj"]`。
改完后参数是 3,149,314——正好是 LoRA 的 2 倍，符合 `init_r=32` vs `r=16`。

**坑二：`update_and_allocate()` 必须每步手动调用，HF Trainer 不会替你调。**
peft 的 docstring 写得很清楚：「should be called in every training step after
`loss.backward()` and before `zero_grad()`」。不调用的后果不是报错：
rank 裁剪从未发生，而 `AdaLoraModel.forward` 里的正交正则项
（`orth_reg_weight` 默认 0.5）却一直加在 loss 上——
起始 loss 是 **2.34** 而不是 ln2≈0.693。

解法是挂个回调（`peft_trainer.build_adalora_callback`），
`on_pre_optimizer_step` 正好是「反向传播完、参数更新前」这个时点：

```python
class AdaLoraScheduleCallback(TrainerCallback):
    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        model.base_model.update_and_allocate(state.global_step)
```

**坑三：沿用 LoRA 的学习率学不动。**
AdaLoRA 的前向里增量被**除以 ranknum**：

```
result += x @ (A * E).T @ B.T * scaling / (ranknum + 1e-5)
```

`init_r=32` 时等效步长比 LoRA 小一个数量级以上。LoRA 用 1e-4，
AdaLoRA 要 **5e-4** 才动得起来。

**还有一个会让人误判的现象：第一个 epoch 看起来是死的。**

```
epoch 1:  val_acc 0.5256   ← 像是又失败了
epoch 2:  val_acc 0.9504   ← 裁剪调度走完后突然起飞
train_loss: 2.12 → 0.71（卡了大半个 epoch）→ 0.206
```

前期模型的容量都花在压正交正则和调整 rank 分配上，分类几乎没学。
**不要看到第一个 epoch 是 0.52 就掐掉**——这是 AdaLoRA 的正常形态，
和 LoRA 第一个 epoch 就有 0.955 完全不同。

> ⚠️ peft 0.15 起 `AdaLoraConfig` 必须传 `total_step`，否则直接 `ValueError`。
> demo 代码写于旧版本，没有这个参数。

### 2.3 P-Tuning（Liu et al., 2021）

思路完全不同：**一个权重都不改**，只在输入 embedding 前面拼 20 个虚拟 token。

本质是「写提示词」的连续化版本。人写的 prompt（"这条影评的情感是：__"）
受词表限制、且只能手工试；虚拟 token 是连续向量，可以直接用梯度优化，
表达空间大得多。

关键设计是这些向量**不直接优化**，而是由一个小 encoder（MLP 或 LSTM，
`encoder_hidden_size=128`）生成：

```
[p_1, ..., p_20] = PromptEncoder(20 个可训练索引)
```

为什么绕这一层：直接优化 20 个互不相关的 embedding 时，优化面非常崎岖，
小模型上经常收敛不了、方差极大。套一个共享 encoder 把参数绑在一起后训练稳定得多。
训完 encoder 可以丢掉，只留生成好的 20 个向量。

可训练参数是四种方法里最少的（本项目 0.125% 级别）。

### 2.4 Prefix-Tuning（Li & Liang, 2021）—— 以及它为什么换了底座

P-Tuning 只在输入层动手，Prefix-Tuning 在**每一层**的注意力里都拼前缀，
而且拼的是 K 和 V：

```
K' = [P_k ; K]        V' = [P_v ; V]
```

真实 token 算注意力时能"看到"这些前缀，相当于每层都注入一段可学习的任务指令。
和 P-Tuning 正好构成一组消融：**只插输入层** vs **每层都插**。

**但它在 DeBERTa 上跑不了**，这是本阶段最实在的一个发现：

peft 的实现是把前缀伪装成 `past_key_values`（KV cache）喂进模型，
这要求底座实现 KV cache。DeBERTa 是纯 encoder，`DebertaV2Model` 没有这个入口，
实跑直接抛：

```
ValueError: Model does not support past key values which are
            required for prefix tuning.
```

而且 peft 还额外禁止它和 gradient checkpointing 同时开：

```
ValueError: PREFIX_TUNING does not work with gradient checkpointing.
```

道理是一样的——checkpoint 段反向重算时，那份 KV cache 已经不在计算图里了。

所以 Prefix-Tuning 换到 `roberta-base`（支持 `past_key_values`，实测可跑），
在对比表里作为**带脚注的补充行**，不与另外三种直接比准确率：底座不同，比了不公平。

> **这本身就是结论**：方法的可用性受底座架构限制。
> **LoRA 系几乎通吃**（只要有线性层就能挂），**prompt 系挑架构**。
> 这也是工程上大家默认选 LoRA 的一个隐性理由。

---

## 3. 显存怎么省下来的：三招叠加

导师在评审里点名的三件事，本质是三个独立的旋钮，可以叠乘。

### 3.1 gradient checkpointing —— 用算力换显存

反向传播需要用到前向时每一层的输出（activation）。默认做法是**全存下来**，
所以显存随「层数 × batch × 序列长度」线性增长。

gradient checkpointing 只保留少数几个"检查点"，其余的在反向传播需要时
**从最近的检查点重新前向算一遍**。

> 类比：不背整本书，只记章节标题，用到某页时从章首重推。

对 L 层的网络，显存从 O(L) 降到约 O(√L)，代价是多一次前向，实测慢 20~35%。

**这里有个必踩的坑**。PEFT 把底座整个冻结后，第一层的输入 `requires_grad=False`，
checkpoint 段会判断「这段不需要梯度」而根本不建计算图，反向传播直接报

```
RuntimeError: element 0 of tensors does not require grad and
              does not have a grad_fn
```

解法是显式让输入需要梯度：

```python
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()      # ← 少了这行就报上面的错
```

`use_reentrant=False` 也不是可选项：reentrant 版本和冻结参数、和 Trainer 对
`use_cache` 的处理都容易打架，PyTorch 2.x 明确推荐非 reentrant 实现。

### 3.2 batch size —— 唯一直接压 activation 的旋钮

batch size 就是「一次同时看几条影评」。模型要为这一批里的每条样本都存中间结果，
所以 activation 显存 ∝ batch size。**它也是唯一能压 activation 的旋钮**
（16P 那部分跟它无关）。

但它会影响训练效果：batch 太小，每步的梯度是在很少的样本上估的，方向"飘",
训练不稳、准确率可能掉——所以不能无限调小。

### 3.3 gradient accumulation —— 把调小的 batch 补回来

导师的原话是「类似于采取一个高 batch size 来算」。做法是**攒够了再更新**：

```
真实 batch 8 × 累积 4 步 = 等效 batch 32

看 8 条 → 算梯度，累加进缓冲区，不更新
看 8 条 → 算梯度，继续累加
看 8 条 → 算梯度，继续累加
看 8 条 → 算梯度，四份攒够 → 更新一次参数，清空缓冲区
```

显存只需装 8 条的量，梯度的统计效果约等于 batch 32。代价是慢：
要跑 4 次前向 + 反向才更新一次参数。

**梯度缓冲区不额外占显存**：它和参数的梯度张量是同一块内存，本来就要分配。

本项目把「等效 batch」硬性锁在 32，和阶段二的 11 个模型完全一致——
否则准确率的差异分不清是方法带来的还是 batch 带来的，对比表就失去意义。
代码里 `--grad-accum` 留空时自动取 `32 // batch_size`。

### 3.4 附带一招：group_by_length

IMDB 影评长度方差极大（中位数 177 词，最长几千）。默认随机组 batch 时，
一条长评论会把整个 batch 都 padding 到它的长度，白算一大片 `[PAD]`。
`group_by_length=True` 把长度相近的样本放进同一个 batch，padding 大幅减少——
这是变长输入下压低显存峰值最便宜的一招，不影响任何超参数。

---

## 4. 绝不让显存爆炸：mem_guard.py

单卡试错最怕两件事：一是 OOM 把同卡的其他进程一起带走，
二是主机内存被吃光触发 swap，整机卡死到连 Ctrl-C 都按不进去
（比 CUDA OOM 危险得多——那个至少能捕获）。

`core/mem_guard.py` 做三层保险：

| 函数 | 作用 | 保护谁 |
|---|---|---|
| `cap_gpu(13.5)` | `set_per_process_memory_fraction`，超限抛可捕获的 CUDA OOM | 同卡的其他进程 |
| `MemoryGuard` 回调 | 每 10 步查显存峰值和进程 RSS，越线主动停训 | 主机内存（PyTorch 管不到） |
| `peak_report()` | 把峰值写进 summary | 让「用了多少」变成可引用的数字 |

上限设 13.5 GB / 15.36 GB：留约 1.5 GB 给 CUDA context、cuDNN workspace 和碎片。
实测顶到 14.5 GB 以上时，失败方式会从「干净的异常」退化成驱动层报错、进程 kill 不掉。

被看门狗中止的运行**不写 summary**（`core/peft_trainer.py` 检查
`trainer.state.mem_guard_aborted`），绝不让一次半截的训练冒充正常实验混进对比表。

---

## 5. 试点：先探显存，再正式跑

正式训练一次要一小时。如果显存不够，它可能在第 40 分钟才爆——白等，还可能拖死机器。
所以放大之前先探针，`--probe-steps N` 只跑 N 步、不做验证不落盘，
读一次显存峰值就退出（**峰值在前十几步就稳定了，之后基本是平的**）。

实测数据（`max_length=384`，gradient checkpointing 开，看门狗上限 13.5 GB）：

| 底座 | 参数量 | 真实 batch × 累积 | 显存峰值 | 速度 | 用了吗 |
|---|--:|---|--:|---|:-:|
| deberta-v3-base | 1.85 亿 | 4 × 8 | 0.88 GB | — | 冒烟测试 |
| deberta-v3-large | 4.37 亿 | 8 × 4 | 2.48 GB | 慢 2 倍 | |
| deberta-v3-large | 4.37 亿 | 16 × 2 | 3.24 GB | | |
| **deberta-v3-large** | **4.37 亿** | **32 × 1** | **4.80 GB** | ~30 min/epoch | ✅ 主实验 |
| deberta-v2-xxlarge | **15.7 亿** | 2 × 16 | **3.49 GB** | ~1.9 h/epoch | ❌ 太慢 |

两个值得记下的点：

1. **xxlarge 真的装得下**。15.7 亿参数、fp16 权重 + LoRA + gradient checkpointing，
   峰值只有 3.49 GB——不到 T4 的 1/4。全量微调它需要约 25 GB。
   没用它做主表**纯粹是因为慢**（一个方法要近 4 小时），不是因为跑不动。
2. batch 32 不需要梯度累积，等效批大小天然就是 32，正好对齐阶段二的口径，
   而且比 batch 8 + 累积 4 快一倍（GPU 利用率更高）。

> `--load-fp16` 只在 xxlarge 这档才用，而且必须配合「把可训练参数转回 fp32」：
> 否则 LoRA 参数也是 fp16，`GradScaler` 会拒绝——
> `ValueError: Attempting to unscale FP16 gradients`。
> 混合精度的 master weight 必须是 fp32，这是 AMP 的硬要求。

---

## 5.5 两个「不报错但结果是错的」的坑，以及怎么兜住它们

阶段三最贵的两个 bug 都不报错，而且都指向同一件事：
**能跑通、格式正常、数字看起来合理，都不等于对。**

**坑 A：`group_by_length` 让提交文件逐行错位。**
开这个开关是为了减少 padding、压低显存峰值，它确实有效。但
`Trainer._get_eval_sampler` 里有一句：

```python
if self.args.group_by_length:
    return LengthGroupedSampler(...)      # 不是 SequentialSampler！
```

也就是说它**也作用于预测**。`predict()` 返回的概率是「按长度分组后」的顺序，
和按文件原序的 `id` 配对，就是彻底的逐行错位。

诊断这个 bug 的关键线索是一个**看似矛盾的组合**：

```
验证集准确率  0.9570      ← 很正常
测试集 AUC    0.5021      ← 等于随机
```

验证集不受影响，因为 `compute_metrics` 拿到的 logits 和 labels 是同一个置换顺序，
一一对应。**只要看到「验证集很高但提交文件是随机」，先查预测顺序。**

**坑 B：DeBERTa 的 pooler 没被保存，adapter 重载后失效。**
`DebertaV2ForSequenceClassification` 的结构是
`encoder → pooler.dense(1024×1024) → classifier(1024→2)`，
而 pooler 和 classifier **在 `from_pretrained` 时都是随机初始化的**
（日志里那句 `newly initialized: classifier.*, pooler.dense.*`）。

peft 的 SEQ_CLS 只自动把 `classifier` 放进 `modules_to_save`。pooler 于是既不训练、
也不保存：训练时它固定在一份随机权重上，LoRA 学着去配合**那个特定的随机投影**；
重新加载时又生成**另一个**随机 pooler，学到的方向全部失效——

```
验证集（训练进程内）  0.9570
重载 adapter 后        0.4417   AUC 0.3116   ← 比瞎猜还差，反相关
```

修法是 `modules_to_save=["pooler"]`。它同时也**应该**被训练：
让一个随机初始化的 1024×1024 投影全程冻结本来就没有道理。

**兜住它们的三条措施**（都已落地）：

1. `--subset` **不再截断测试集**。之前截断成 1000 行，冒烟测试根本没法拿公开标签
   核对提交文件，这类错位只能等跑完整套才暴露。多花几分钟推理换一个真能兜住它的测试；
2. `tools/score_submissions.py`：直接给已有的提交 CSV 打分，不重建模型。
   打分和训练解耦，一分钟就能验证一份提交文件到底对不对；
3. `tests/test_peft.py` 里各有一条回归测试——一条钉住「predict 时
   `group_by_length` 必须是 False」，一条钉住「LoRA 系配置必须包含 pooler」。

## 6. 踩过的坑汇总

按遇到的顺序，全部是实跑出来的：

| # | 报错 / 现象 | 原因 | 解法 |
|---|---|---|---|
| 1 | `ValueError: Attempting to unscale FP16 gradients` | `torch_dtype=float16` 让 LoRA 参数也成了 fp16，AMP 的 GradScaler 拒绝 unscale | 默认 fp32 加载；确需 fp16 权重时把可训练参数 `.float()` 转回来 |
| 2 | `PREFIX_TUNING does not work with gradient checkpointing` | 前缀经 `past_key_values` 注入，checkpoint 反向重算时那份缓存不在图里 | Prefix 自动关掉 checkpointing，并在表里注明 |
| 3 | `Model does not support past key values` | DeBERTa 是纯 encoder，没有 KV cache | Prefix 换 `roberta-base`，作为补充行 |
| 4 | `element 0 of tensors does not require grad` | 底座全冻结后 checkpoint 段不建图 | `enable_input_require_grads()` + `use_reentrant=False` |
| 5 | AdaLoRA `ValueError`（缺 total_step） | peft 0.15 起裁剪计划必须知道总步数 | 由 `epochs × steps_per_epoch` 算出来传进去 |
| 5a | AdaLoRA 准确率 0.5094（瞎猜），loss 卡在 ln2 | ①默认挂载点比 LoRA 宽 6 倍 ②`update_and_allocate` 从未被调用 ③lr 太低 | 写死 `target_modules`、加 `on_pre_optimizer_step` 回调、lr 提到 5e-4。三个都不报错，见 2.2 节 |
| 5b | Prefix-Tuning 只有 0.8546，像是方法不行 | 2 epochs 根本没收敛（结束时 loss 还在 0.41，P-Tuning 是 0.138） | 加到 6 epochs + lr 3e-4 → 0.8976，第 5-6 轮曲线压平才算收敛 |
| 6 | 找不到目标模块 | demo 注释里的 `['q_proj','v_proj']` 是 LLaMA 的命名 | 留空，用 peft 对 DeBERTa 的默认映射 `query_proj`/`value_proj` |
| 7 | 学不动（loss 几乎不降） | 沿用全量微调的 lr=2e-5 | PEFT 只训 0.x% 参数，梯度信号弱，lr 提到 1e-4 |
| 8 | `./result/` 目录不存在 | demo 的笔误，且跑到最后一步才崩 | 统一 `results/`（和阶段二同一个坑） |
| 9 | `evaluation_strategy` TypeError | transformers 4.46 起改名 `eval_strategy` | 改名 |
| 10 | `prepare_model_for_int8_training` ImportError | 已从 peft 移除（改叫 `prepare_model_for_kbit_training`） | 本项目不做 int8，直接删掉 |
| **11** | **提交文件分数等于随机（AUC 0.5021），但验证集有 0.9570** | `group_by_length=True` **也作用于 predict**：`_get_eval_sampler` 返回 `LengthGroupedSampler`，概率按长度重排后与按文件原序的 id 逐行错位 | predict 前先 `trainer.args.group_by_length = False`（见 `predict_in_order`）。**这个组合就是它的指纹**——验证集不受影响，因为 logits 和 labels 同序 |
| **12** | **重载 adapter 后测试集 0.4417 / AUC 0.3116（反相关）** | DeBERTa 的 `pooler.dense`(1024×1024) 是随机初始化的，peft 只自动保存 `classifier`，pooler 既不训练也不保存；重载时 `from_pretrained` 生成**另一个**随机 pooler | `modules_to_save=["pooler"]`。顺带也该训它——让随机初始化的投影全程冻结本来就没道理 |

第 1、2、3、4 条是这一阶段新踩的；8、9、10 是 demo 里和阶段二一模一样的老坑
（见 [troubleshooting.md](troubleshooting.md)）。

---

## 7. 怎么跑

```bash
# 探显存（20 步，几十秒，不落盘）
python experiments/peft/lora.py --probe-steps 20

# 正式训练（默认 deberta-v3-large, batch 32, 2 epochs, ~1 小时）
python experiments/peft/lora.py
python experiments/peft/adalora.py
python experiments/peft/ptuning.py
python experiments/peft/prefix.py           # 默认 roberta-base，见 2.4

# 显存不够就调小 batch，累积步数自动补到等效 32
python experiments/peft/lora.py --batch-size 8

# 复现「xxlarge 也能跑」这个结论
python experiments/peft/lora.py --model-id microsoft/deberta-v2-xxlarge \
    --batch-size 2 --load-fp16 --probe-steps 8

# 汇总进对比表和 README
python tools/collect_results.py
```

## 8. 参考

- LoRA: *LoRA: Low-Rank Adaptation of Large Language Models*, Hu et al., 2021 — [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- AdaLoRA: *Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning*, Zhang et al., 2023 — [arXiv:2303.10512](https://arxiv.org/abs/2303.10512)
- Prefix-Tuning: *Optimizing Continuous Prompts for Generation*, Li & Liang, 2021 — [arXiv:2101.00190](https://arxiv.org/abs/2101.00190)
- P-Tuning: *GPT Understands, Too*, Liu et al., 2021 — [arXiv:2103.10385](https://arxiv.org/abs/2103.10385)
- Gradient checkpointing: *Training Deep Nets with Sublinear Memory Cost*, Chen et al., 2016 — [arXiv:1604.06174](https://arxiv.org/abs/1604.06174)
- 官方示例：[huggingface/peft — examples/sequence_classification](https://github.com/huggingface/peft/tree/main/examples/sequence_classification)
