# DeBERTa-v3-large + LoRA + 监督对比 / MoCo

阶段四。四个运行，真实 batch=4，effective batch 32，seed 42，
其余超参与 `16_lora` 那次（0.9633）完全一致，只差方法本身与动量系数。

| 提交文件 | 方法 | 测试集准确率 | ROC-AUC | 训练耗时 | 显存峰值 |
| --- | --- | --: | --: | --: | --: |
| `baseline_bs4_seed42_submission.csv` | 纯交叉熵 | 0.9629 | 0.9924 | 3936 s | 2.19 GB |
| `scl_bs4_seed42_submission.csv` | + 监督对比（双视图） | 0.9635 | 0.9918 | 7593 s | 2.25 GB |
| `m0999_bs4_seed42_submission.csv` | + 队列与动量编码器，m=0.999 | 0.9627 | 0.9914 | 5150 s | 2.22 GB |
| `m099_bs4_seed42_submission.csv` | + 队列与动量编码器，m=0.99 | 0.9633 | 0.9924 | 5131 s | 2.22 GB |

四组极差 0.08 个百分点（25,000 条里 20 条），McNemar p 全部大于 0.27，分不出差别。
每组一个 seed。实现、验证与结论见 [../../experiments/moco/README.md](../../experiments/moco/README.md)，
完整数据见 [../../results/scl_moco_comparison.md](../../results/scl_moco_comparison.md)。

每份都是 25,000 行正面情感概率，格式与其余提交文件一致。
