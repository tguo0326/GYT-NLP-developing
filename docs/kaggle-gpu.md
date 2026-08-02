# Kaggle 免费算力：怎么用，以及这个项目为什么不用

## 一、Kaggle 提供什么

Kaggle Notebook（Kernels）对已验证手机号的账号免费开放加速器。以本文写作时的
公开文档为准：

| 资源 | 额度 | 说明 |
|---|---|---|
| CPU Notebook | 不限时长（单次会话有上限） | 约 4 核 / 30 GB 内存 |
| GPU | **每周 30 小时**（滚动重置） | 型号随平台调整，常见 P100 / T4 ×2 |
| TPU | **每周 20 小时** | v3-8，需要 TensorFlow / JAX 或 PyTorch-XLA |
| 单次会话上限 | 交互 12 小时 / 后台提交 9 小时 | 超时会被强制中断 |
| 磁盘 | `/kaggle/working` 约 20 GB | 只有这个目录可写且能持久化 |

> 额度和型号会随平台策略变化，**以 Notebook 右侧边栏实时显示的数字为准**。
> 网上流传的「无限 GPU」「绕过配额」的方法违反 Kaggle 服务条款，会导致封号，
> 且本项目完全不需要。

## 二、开启 GPU 的步骤

1. 账号需先在 Settings → Phone Verification 完成手机验证，否则看不到加速器选项。
2. 打开 Notebook → 右上角 **⋮** → **Accelerator** → 选 GPU T4 ×2 或 GPU P100。
3. 侧边栏会显示本周剩余额度。
4. 验证是否真的挂上了：

```python
import torch
print(torch.cuda.is_available(), torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
```

## 三、这个项目为什么全程用 CPU

这不是保守，而是三个部分的计算特征决定的：

| 阶段 | 主要计算 | GPU 有用吗 |
|---|---|---|
| Part 1 · CountVectorizer | 哈希 + 稀疏矩阵计数 | ❌ 纯字符串处理，无稠密矩阵乘法 |
| Part 1/3 · RandomForest | 决策树分裂，大量分支判断 | ❌ scikit-learn 无 GPU 后端；树模型分支多，不适合 SIMD |
| Part 2 · gensim Word2Vec | Cython + 多线程 SGD | ❌ gensim 没有 GPU 实现，靠 `workers` 吃多核 |
| Part 3 · MiniBatchKMeans | 稠密矩阵距离计算 | ⚠️ 理论上可加速，但本项目规模（1.6 万 × 300）CPU 只要几秒 |

实测：整套流程在 4 核 CPU 上跑完约 10 分钟，其中 Word2Vec 训练 5 个 epoch
不到 1 分钟。**开 GPU 一秒钟都省不下来，只会白扣每周 30 小时的额度。**

一个常见误解：「深度学习/NLP 就要用 GPU」。GPU 快是因为它擅长大规模并行的
稠密浮点运算。上表里没有一个阶段是这种负载——瓶颈在字符串处理和树的分支判断，
这些都是 CPU 的活。

## 四、什么时候该开 GPU

沿着这个项目往下走，到这些场景就该开了：

- 用 PyTorch / TensorFlow 训练 LSTM、CNN 文本分类器；
- 微调 BERT / RoBERTa 做情感分类（**这一步 GPU 能带来数十倍差距**）；
- 调用 `sentence-transformers` 批量生成句向量；
- 任何涉及 Transformer 前向/反向传播的任务。

判断标准很简单：**主要计算是不是大批量的稠密矩阵乘法。** 是就开，不是就别开。

## 五、省额度的实用技巧

1. **先用小样本调通流程再全量跑。** 前期 bug 全在 CPU 会话里修完，
   等代码确认无误再切 GPU 跑最终版本。
2. **用 Save Version（Save & Run All）跑长任务。** 后台执行，浏览器可以关掉，
   不占交互会话时长。
3. **不用 GPU 时立刻关掉会话。** 侧边栏 Stop Session；否则挂着也在计时。
4. **把中间产物存成 Dataset。** 比如 Part 2 训练好的词向量存成 Kaggle Dataset，
   Part 3 直接 Add Input 挂载，不必每次重训。
5. **注意 `/kaggle/working` 20 GB 上限。** 只有这个目录能持久化，
   `/kaggle/input` 是只读的。
6. **`isInternetEnabled` 默认关闭。** 需要 `pip install` 或下载模型时，
   要在 Settings 里手动打开网络（打开后该 Notebook 不能用于部分竞赛提交）。

## 六、把本项目跑在 Kaggle 上

```
1. 打开 https://www.kaggle.com/competitions/word2vec-nlp-tutorial
   点 Join Competition（不接受规则会看不到数据）
2. Code → New Notebook → File → Import Notebook
   上传 notebooks/part1-bag-of-words.ipynb
3. 右侧 Add Input → Competitions → 搜 word2vec-nlp-tutorial → Add
4. Accelerator 保持 None
5. Run All
6. Save Version → 生成的 /kaggle/working/submission.csv
   可在 Output 标签页 Submit to Competition
```

Part 2 会在 `/kaggle/working/` 生成 `word2vec_300d.model`。要在 Part 3 里复用，
把它从 Output 存成一个 Dataset，再在 Part 3 的 Notebook 里 Add Input 挂载即可。

## 参考

- [Kaggle Notebooks 文档](https://www.kaggle.com/docs/notebooks)
- [Kaggle 加速器说明](https://www.kaggle.com/docs/efficient-gpu-usage)
