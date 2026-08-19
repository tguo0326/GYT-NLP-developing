# 监督对比学习 + MoCo（SCL-MoCo）

在 `experiments/peft/lora.py` 那套 DeBERTa-v3-large + LoRA 之上，加两件东西：

1. **监督对比损失**（SupCon / SCL）：交叉熵只管分类边界画对没有，不管特征空间长什么样。
   SCL 额外要求同标签的句向量互相靠近、不同标签的互相推远。
2. **MoCo 的队列与动量编码器**：SCL 的正负样本只能从当前 batch 里凑，batch 小了就凑不够。
   MoCo 把过去若干步的特征存进一个固定长度的队列当候选，并且用一个 EMA 更新的
   动量编码器来产生这些特征，保证队列里的特征彼此一致。

想验证的是一句判断：**真实 batch 小的时候普通 SCL 会退化，队列能把它救回来。**
结论是在这份数据上没有观察到这个现象，四组准确率分不出差别，
但过程中确认了一件更有用的事 —— 动量系数必须按训练步数折算，照搬论文的 0.999 会让
动量分支基本失效。完整数据见 [../../results/scl_moco_comparison.md](../../results/scl_moco_comparison.md)。

## 结果速览

真实 batch=4，effective batch 32，seed 42，其余超参与 0.9633 那次完全一致。

| 方法 | 测试集 Accuracy | ROC-AUC | 每 query 候选数 | q·k | 对比目标完成度 | 训练时间 |
| --- | --: | --: | --: | --: | --: | --: |
| Baseline（纯 CE） | 0.9629 | 0.9924 | — | — | — | 66 min |
| SCL | 0.9635 | 0.9918 | 4 | 0.998 | 87% | 127 min |
| SCL-MoCo m=0.999 | 0.9627 | 0.9914 | 4100 | 0.515 | 21% | 86 min |
| SCL-MoCo m=0.99 | 0.9633 | 0.9924 | 4100 | 1.000 | 79% | 86 min |


> `q·k` 与「对比目标完成度」这两个诊断量是跑完 m=0.999 那轮之后才加进代码的，
> 所以 SCL 与 m=0.999 两行的数值是事后按公式从已落盘的 `view_gap` 和对比损失换算的，
> 只有 m=0.99 那行是训练时直接记录进 summary 的。换算口径见 `trainer_scl._record_progress`。

参照：同一套超参、真实 batch=32 的 `deberta_lora` 是 0.9633 / 0.9924。
真实 batch 从 32 降到 4（effective batch 不变）对纯交叉熵几乎没有影响 ——
差 10 条样本，所以「小 batch 有害」这个前提本身在这份数据上就很弱。

四组极差 0.08 个百分点（25,000 条里 20 条），McNemar p 全部大于 0.27，
预测概率之间的相关系数都在 0.99 以上。每组只有一个 seed。

## 文件

| 文件 | 作用 |
| --- | --- |
| `data.py` | 复刻 `core/peft_trainer.py` 的读取与划分：`QUOTE_NONE`、`test_size=0.2, random_state=42, stratify`、测试集保持原序、公开标签按 id → 正文两级对齐 |
| `losses.py` | 官方 `SupConLoss`（只作数值基准）、`supcon_core`（SCL 与 SCL-MoCo 共用的同一套数学）、`scl_two_view`、逐步统计 |
| `moco.py` | `FeatureQueue`（特征 + 标签 + 指针 + 有效长度，全是 buffer）、`MomentumBranch`（EMA、硬同步、no_grad 前向）、`ProjectionHead` |
| `model.py` | DeBERTa 冻结底座 + LoRA + `modules_to_save=["pooler","proj_head"]`；SCL-MoCo 再挂第二套 adapter |
| `trainer_scl.py` | `SCLMoCoTrainer`，三种方法共用一条代码路径；EMA 挂在 `on_optimizer_step` |
| `run_experiment.py` | 单组实验入口，默认值就是 0.9633 那次的配置 |
| `smoke_test.py` | 22 项真实性断言，跑正式实验之前必须先过 |
| `collect_results.py` `significance.py` `plot_curves.py` `progress.py` | 汇总表、McNemar 检验、曲线、训练中查进度 |
| `run_stage1.sh` `run_stage2.sh` | 复现脚本，batch=4 / batch=16 各三组 |

## 跑法

```bash
cd experiments/moco

python smoke_test.py                         # 22 项断言，约 4 分钟，不过就别往下跑
./run_stage1.sh                              # batch=4 三组，约 5 小时
BS=16 ./run_stage1.sh                        # batch=16 三组（= run_stage2.sh，还没跑）

python run_experiment.py --method scl_moco --batch_size 4 --momentum 0.99
python run_experiment.py --method scl_moco --batch_size 4 --probe_steps 12 --subset 2000
                                             # 只测速度和显存峰值，不落盘

python collect_results.py                    # 对照表
python significance.py                       # McNemar 检验
python plot_curves.py                        # ../../results/scl_moco_curves_bs4.png
python progress.py                           # 训练中看进度与 ETA
```

产物落在仓库约定的位置：`../../results/scl_moco_*`、`../../logs/scl_moco_*.log`、
adapter 与队列状态在 `../../models/`（不入库）。

## 实现

### Query 编码器

`DeBERTa-v3-large（冻结）+ LoRA(query_proj, value_proj) + pooler + ProjectionHead(1024→1024→128)`。
分类路径一点没动，仍然是 `pooler → classifier → 交叉熵`。对比特征走另一条路：
用 forward hook 抓骨干的 `last_hidden_state`，mask 平均池化，过投影头，L2 归一化。

不用 `output_hidden_states=True` 取隐藏状态，那会把 24 层的输出全留在显存里，
把 gradient checkpointing 省下来的又吃回去。

`modules_to_save` 必须带上 `pooler`，理由和 `core/peft_trainer.py` 里一样：
DeBERTa 的 `pooler.dense` 是随机初始化的，不显式加进来它就全程冻结在一份随机投影上。

### 动量 key 编码器

底座本来就是冻结的，所以**没有复制第二份 DeBERTa-v3-large**。做法是在同一个底座上再挂
一套 peft adapter，`modules_to_save` 机制会为每个 adapter 各存一份 pooler / proj_head /
classifier。动量副本一共 3,805,314 个参数，占模型总量的 0.857%，约 14.5 MB；
复制整个底座是 +4.35 亿参数、约 1.7 GB。

- 初始化用硬拷贝，逐张量最大绝对差为 0。peft 给第二个 adapter 的 `lora_A` 是**另一次**
  随机初始化，不显式同步就不满足「两个编码器起点一致」。
- `requires_grad=False`，不进 optimizer，前向全程 `torch.no_grad()`。
- `θ_k ← m·θ_k + (1−m)·θ_q`，挂在 `on_optimizer_step` 回调上。

**EMA 必须跟着 optimizer step，不能跟着 micro-batch。** 梯度累积 8 步时一个优化步包含
8 次前向，每次前向都做一遍 EMA 的话，动量的实际衰减速度就是设定值的 8 倍，
而且 batch=4（累积 8）和 batch=16（累积 2）之间不可比。入队则是每个 micro-batch 都做，
每个 micro-batch 的 key 都是合法特征，没有理由丢掉。

### 特征队列

`queue_size=4096`，存 128 维归一化特征和真实标签，指针、有效长度都是 buffer，
所以 `state_dict()` 天然带上它们，checkpoint 保存恢复不用额外写代码。

初始队列不填随机特征和随机标签：特征全 0、标签全 −1，靠有效长度屏蔽，
只有前 `valid` 个槽位参与对比。队列满了之后 FIFO 覆盖最旧的，跨过队尾时分两段写。

### 正负样本划分：两组的候选结构刻意对齐

|  | anchor | 候选 | 正样本 | 负样本 |
| --- | --- | --- | --- | --- |
| SCL | 视图 1（算交叉熵那次前向） | 视图 2 的全部 B 条，同一个在线编码器再前向一次、dropout 独立重采样 | 自身第二视图 + 同标签样本 | 不同标签样本 |
| SCL-MoCo | query | 动量 key 的 B 条 + 队列 K 条 | 自身动量 key + 同标签样本 | 不同标签样本 |

这样队列为空时，两组每个 query 的候选数、按标签划分的正负数量逐位相同，
两组都不会因为凑不出正样本而跳过 anchor。差异被收窄到唯一一项：
那 B 条来自在线编码器（带梯度、没有队列）还是动量编码器（不带梯度）加历史队列。

一开始 SCL 是单视图的（候选就是 batch 内另外 3 条），smoke test 暴露出 batch=4 时
**约一半的 micro-batch 里会有 anchor 完全没有正样本而被整条丢掉**，
那样它和 SCL-MoCo 之间就多出「有没有配对正样本」这个额外差异，归因不干净。
单视图版本仍留在 `losses.scl_in_batch`，只用来跟官方 `SupConLoss` 对数值。

两条路径调用的是同一个 `supcon_core`，SupCon 论文的 L_out 形式：

```
L_i = -1/|P(i)| · Σ_{p∈P(i)} log[ exp(q_i·c_p/τ) / Σ_{a∈A(i)} exp(q_i·c_a/τ) ]
```

没有正样本的 query 从损失里剔除，不是填 0；整个 batch 都没有正样本时返回与计算图连通的
`query.sum()*0.0`，既不断梯度也不产生 nan。

### 总损失

`L = CE + λ · L_contrastive`，λ=0.2、τ=0.3（沿用之前 SCL 实验的取值）。
三组的交叉熵都只由视图 1 计算。**评测一律纯交叉熵**，否则 `eval_loss` 在三组之间不可比。

## 真实性验证

`smoke_test.py` 22 项断言，原始输出在 [../../logs/scl_moco_smoke_test.log](../../logs/scl_moco_smoke_test.log)。
分三部分：A 是纯 CPU 的损失数学与队列机制，B 是真实模型加真实 Trainer 跑 8 个 optimizer
step，C 是三组方法横向对比。挑几条关键的：

```
A1  supcon_core 与官方 SupConLoss 数值等价    2.90006351 / 2.90006351   差 0.00e+00
A3  (指针, 有效长度) 轨迹 [(4,4),(8,8),(2,10),(6,10)]      第 3 步跨队尾绕回
    槽位 2-3 的特征等于最新那批、不再等于最旧那批            最旧数据确实被顶掉
A4  队列为空时两组逐 query 完全一致（标签 [0,0,1,0]）
    SCL 与 SCL-MoCo 都是 候选 4、正 (3,3,1,3)、负 (1,1,3,1)、4/4 个 query 参与损失
    对照：单视图 SCL 的正样本数是 (2,2,0,2)，第 3 条会被丢掉
B1  104 组参数对、3,805,314 个参数，逐张量最大绝对差 0.000e+00
B2  optimizer 参数 104 个；key 参数落在 optimizer 里的 0 个；query 104/104 全命中
B5  |Δquery|=5.485e-05  |Δkey|=5.402e-08  比值 0.00098（≈1−m）
    θ_k ← m·θ_k+(1−m)·θ_q 的最大偏差 1.863e-09
B6  optimizer step 8 次，EMA 8 次，micro-batch 16 次      EMA 没有按 micro-batch 跑
B10 非零梯度 56 个：query LoRA 48（lora_A 24 + lora_B 24）、proj_head 4、pooler 2
    key 侧带 .grad 的参数 0 个
B11 清零后重新载入，指针 / 有效长度 / 队列特征 / 队列标签 / key / query 逐字节相同
C3  两组各 12 个 micro-batch，跳过的 anchor 累计都是 0
```

B10 有个坑：只看第一个 optimizer step 会误判「梯度没传到 lora_A」——
LoRA 的 B 初始化为 0，第一步 A 的梯度恒等于 0。断言取的是最后一个 step 的梯度。

正式训练里也留了硬断言，`../../logs/scl_moco_m099_bs4_seed42.log` 里能查到：

```
[断言] optimizer 参数 104 个；key 参数 104 个，其中在 optimizer 里的 0 个
[断言] 训练开始时 query 与 key 参数最大绝对差 = 0.000e+00
EMA 调用 1250 次；入队 10000 次共 40000 条；队列有效长度 4096，指针 3136
[断言] EMA 次数 == optimizer step 数 == 1250（不是 micro-batch 数 10000）
```

## 动量系数必须按训练步数折算

这是这轮最实用的一条结论。第一次跑 SCL-MoCo 直接用了论文的 `m=0.999`，
诊断量显示动量分支基本没在工作：

- 同一句话过两个编码器，特征余弦只有 **0.515**。MoCo 里这一对就是要被拉近的正样本对，
  0.5 意味着模型在对齐一个陈旧的目标。
- 对比损失从 8.28 缓慢降到 8.17，看着在收敛。但候选数 4100 时，
  **特征完全塌缩的损失恰好是 log(4100)=8.32**，完美分开的解析下界是 7.63 ——
  8.17 离塌缩值只走了 21%，等于几乎没学。只看损失绝对值会被 log(N) 这个常数骗过去。
- 队列里两类特征的中心余弦相似度 0.9749，用它做线性分类只有 0.729 准确率，
  而模型本身是 0.9627。队列装的是一堆分不开的特征。

算一下就明白：`m=0.999` 的有效平均窗口是 `1/(1−m)=1000` 步，而这里总共只有 1250 步，
窗口占了训练的 80%，训练结束时 key 里还残留 `0.999^1250 = 28.6%` 的初始权重。
MoCo 原文训 20 万步以上，同样的窗口只占 0.5%。

换成 `m=0.99`（窗口 100 步，残留 3.5e-6），只改这一个参数：

| | m=0.999 | m=0.99 |
| --- | --: | --: |
| q·k 余弦 | 0.515 | 1.000 |
| 对比目标完成度 | 21% | 79% |
| 验证集 Accuracy | 0.9556 | 0.9570 |
| 测试集 Accuracy | 0.9627 | 0.9633 |
| 测试集 ROC-AUC | 0.9914 | 0.9924 |

四个指标方向一致，AUC 回到和 Baseline 相同的 0.9924。所以 m=0.999 那轮观察到的
「对比项损害概率排序」是动量分支陈旧造成的，不是对比学习本身的问题。

但修好之后和 Baseline、SCL 仍然分不出差别（p 分别是 0.546 和 0.813）。

两轮都在 step 100 附近出现一次 q·k 塌陷，位置正好是 warmup 结束、学习率冲到峰值的时刻。
随机初始化的 projection head 在这个窗口里方向转得最快，任何滞后的副本都会瞬间变成正交。
区别只在恢复速度。如果还要继续调，度量学习那边有个标准做法值得试：
等特征空间稳定之后再启用队列（XBM 的做法），前若干步只用 batch 内的 key。

## 为什么队列扩大了 1000 倍却没有收益

三个原因，第二个是这轮才看清的。

1. **均衡二分类下正样本本来就不缺。** batch=4 里平均就有 1.5 个同类样本，
   队列补的是「更多」而不是「有无」。
2. **小 batch 伤 SCL 的途径是「凑不出配对正样本」，不是「正样本不够多」。**
   单视图 SCL 在 batch=4 时约一半 micro-batch 会丢掉 anchor，这才是退化的来源；
   而堵住这个途径只需要让同一批输入多过一次网络，用不到队列也用不到动量编码器。
   队列解决的是另一个问题。
3. **天花板效应。** DeBERTa-v3-large 在 IMDB 上已经 96.3%，剩下的错误多半是 384 token
   截断和标签本身的噪声，正则类方法改不了这些。这和之前 LoRA 只训 0.6% 参数、
   本身就是很强的正则那条结论是一致的。

MoCo 原文解决的是自监督实例判别：1 个正样本对 65535 个负样本，负样本全是不同实例，
队列越长越有价值。监督二分类里队列一填满就饱和了 —— 再多的同类样本提供不了新的梯度方向。

## 一个没有解决的口径问题

对比损失里有 `log Σ_候选` 这一项，候选从 4 涨到 4100 会把损失的绝对量级顶上去：
SCL 稳定在 0.94，SCL-MoCo 稳定在 7.83~8.17。同一个 λ=0.2 之下，
**SCL-MoCo 实际施加的正则拉力比 SCL 强约 8 倍**。

第一轮为了只比较方法本身没有单独调 λ，但这意味着 SCL-MoCo 同时变了两件事
（候选变多、有效正则强度变大）。要分离这两者得扫 λ，按量级差折算大约是 0.2/8.7 ≈ 0.023。

## 失败与异常记录

没有 OOM，没有训练失败，没有重跑，没有 nan。峰值显存 2.19~2.25 GB，
远低于 T4 的 15 GB（0.9633 那次 batch=32 是 4.8 GB）。过程中修掉的问题：

1. **smoke A3 的断言一开始写错了**（不是代码错）：我预期绕回后槽位 0-1 被最新一批覆盖，
   实际被覆盖的是槽位 2-3。改成逐元素比对特征张量，比只看标签更硬。
2. **smoke B10 一开始报 IndexError**：`trainer.train()` 结束后梯度已经被
   `zero_grad(set_to_none=True)` 清掉。改成在 `on_pre_optimizer_step` 抓。
3. **训练进度看不见**：Trainer 那行 `{'loss': ...}` 是 `PrinterCallback` 用 `print` 打的，
   stdout 重定向到文件后被缓冲，几十分钟落不了盘。加了 `LogToFileCallback`，只写日志、
   不改任何训练数值。**Baseline 那组是在加这个回调之前启动的**，所以它的日志里没有逐步
   train loss，只能从 summary 取指标；另外三组有。训练数值口径不受影响。
4. **日志里的「完成度」是用窗口内最后一个 micro-batch 算的**，单 batch 波动大，
   会看到 79% 和 −25% 交替跳。summary 里那个是按窗口平均损失算的，才是稳定值。
5. **中途改过实验定义**：SCL 从单视图改成双视图（见上面「正负样本划分」）。
   改完重跑了全部 22 项 smoke test 才启动正式实验。正式实验每组只跑过一次，
   没有任何一组被丢弃。

## 与 0.9633 的可比性

|  | `deberta_lora`（0.9633 那次） | 本目录的 Baseline |
| --- | --- | --- |
| 指标口径 | 24,961 条公开标签行上的 accuracy | 同 |
| 测试集 Accuracy / AUC | 0.9633 / 0.9924 | 0.9629 / 0.9924 |
| 验证集 Accuracy | 0.9570 | 0.9560 |
| 可训练参数 | 2,624,514 | 2,624,514 |
| 真实 batch / 累积 / 等效 | 32 / 1 / 32 | 4 / 8 / 32 |
| 其余超参 | — | 全部相同 |

Baseline 与那次只差「真实 batch 32 → 4」一项，结果差 0.04 个百分点（10 条样本），
可以认为是同一水平，所以这四组之间的比较是干净的。

两点边界：0.9633 那次没有投影头、真实 batch 是 32，不能当成第五组混进表里；
PEFT 的 checkpoint 不能事后重载再打分（`pooler.dense` 随机初始化不可复现，
原项目实测重载后掉到 0.4417、AUC 0.3116），所以四组的概率都是训练进程内产出的。

## 论文

- He et al., *Momentum Contrast for Unsupervised Visual Representation Learning*, CVPR 2020 — https://arxiv.org/abs/1911.05722
- Khosla et al., *Supervised Contrastive Learning*, NeurIPS 2020 — https://arxiv.org/abs/2004.11362
- Gunel et al., *Supervised Contrastive Learning for Pre-trained Language Model Fine-tuning*, ICLR 2021 — https://arxiv.org/abs/2011.01403
- Wang et al., *Cross-Batch Memory for Embedding Learning*, CVPR 2020 — https://arxiv.org/abs/1912.06798
