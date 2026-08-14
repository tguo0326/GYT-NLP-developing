"""阶段三：参数高效微调（PEFT）——LoRA / AdaLoRA / Prefix-Tuning / P-Tuning 的共用实现。

四个脚本 `experiments/peft/lora.py` / `_adalora.py` / `_prefix.py` / `_ptuning.py`
只有「用哪个 PeftConfig」不同，所以实现收在这里，和阶段二的 `core/hf_trainer.py` 同一套路。

**和阶段二全模型微调的本质区别**：BERT/RoBERTa 微调时 1 亿多个参数全都要更新，
优化器还要为每个参数存两份状态（Adam 的一阶、二阶动量），显存开销是权重的 3 倍。
PEFT 把底座整个冻住，只训练额外插入的一小块：

- **LoRA**：给注意力的 Q/V 投影各挂一对低秩矩阵 `B(d×r) @ A(r×d)`，r=16 时
  一层只多 2·d·r 个参数。原权重 `W` 不动，前向变成 `Wx + (α/r)·BAx`。
  训完可以把 `BA` 合并回 `W`，推理零额外开销——这是它比 Adapter 更受欢迎的原因。
- **AdaLoRA**：LoRA 的每层 rank 都写死成同一个 r，但实际上不同层需要的容量不同。
  AdaLoRA 用 SVD 形式参数化增量，训练中按奇异值的重要性动态裁剪 rank，
  把预算挪给更需要的层。
- **Prefix-Tuning**：不碰权重，在每一层注意力的 K/V 前面拼 20 个可训练的虚拟向量。
  依赖模型支持 `past_key_values`。
- **P-Tuning**：只在输入层前面拼虚拟 token，且这些 token 由一个小 LSTM/MLP
  （prompt encoder）生成而非直接优化——直接优化 embedding 在小模型上很难收敛。

导师在评审里点名的三件事，全部落在这里：

1. `--gradient-checkpointing`（默认开）：不在前向时保存每层的 activation，
   反向传播需要时重算。显存换算力，是 LoRA 能在小卡上跑大模型的另一半原因；
2. `--batch-size` 调小到显存装得下为止；
3. `--grad-accum` 把小 batch 攒够再更新一次参数，等效 batch 仍是 32——
   和阶段二的 11 个模型保持同一口径，否则准确率的差异分不清是方法还是 batch 带来的。

另外修掉 `new work/` 那四份 demo 里会直接崩掉的地方（和 hf_trainer.py 同样的坑）：
`./result/` 目录不存在、`evaluation_strategy` 在 4.46 起改名 `eval_strategy`、
`Trainer(tokenizer=)` 已废弃、`prepare_model_for_int8_training` 已从 peft 移除、
`train_test_split` 缺 `random_state` 和 `stratify` 导致跨模型不可比。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
# 缓解显存碎片。PEFT 训练里 activation 的形状随 batch 内最长样本变化，
# 固定大小的 block 复用率低，expandable_segments 能明显压低峰值。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from core import common
from core import mem_guard

# 与阶段二统一的等效批大小。真实 batch 再小，乘上累积步数都要回到这个数。
EFFECTIVE_BATCH = 32

METHODS = ("lora", "adalora", "prefix", "ptuning")

METHOD_LABELS = {
    "lora": "LoRA",
    "adalora": "AdaLoRA",
    "prefix": "Prefix-Tuning",
    "ptuning": "P-Tuning",
}


def build_parser(*, method: str, model_id: str = "microsoft/deberta-v3-large",
                 batch_size: int = 32, lr: float = 1e-4,
                 epochs: float = 2) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default=method, choices=METHODS,
                        help=argparse.SUPPRESS)
    parser.add_argument("--model-id", default=model_id)
    parser.add_argument("--epochs", type=float, default=epochs)
    parser.add_argument("--batch-size", type=int, default=batch_size,
                        help="每步真实批大小。显存不够就调小，用 --grad-accum 补回来")
    parser.add_argument("--grad-accum", type=int, default=0,
                        help=f"梯度累积步数。默认自动取 {EFFECTIVE_BATCH}//batch_size，"
                             "使等效批大小与阶段二一致")
    parser.add_argument("--max-length", type=int, default=384,
                        help="截断长度。阶段二用 256；这里放到 384 覆盖更多长评论，"
                             "PEFT 省下的显存正好花在序列长度上")
    # LoRA 只训 0.x% 的参数，梯度信号比全模型微调弱得多，学习率要高 1~2 个数量级
    # （全模型微调用 2e-5，这里 1e-4）。沿用 2e-5 会几乎学不动。
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--rank", type=int, default=16, help="LoRA / AdaLoRA 的秩 r")
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--virtual-tokens", type=int, default=20,
                        help="Prefix-Tuning / P-Tuning 的虚拟 token 数")
    parser.add_argument("--seed", type=int, default=common.SEED)
    parser.add_argument("--fp16", action="store_true", default=torch.cuda.is_available(),
                        help="混合精度训练。T4 不支持 bf16，只能用 fp16")
    parser.add_argument("--load-fp16", action="store_true",
                        help="连权重也用 fp16 加载，主机内存和显存都砍半。"
                             "只在底座大到 fp32 装不下时用（如 deberta-v2-xxlarge）")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True,
                        help="默认开启（导师明确要求）")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                        action="store_false")
    parser.add_argument("--gpu-limit", type=float, default=mem_guard.GPU_LIMIT_GB,
                        help="显存硬上限（GB），超过立即中止")
    parser.add_argument("--ram-limit", type=float, default=mem_guard.RAM_LIMIT_GB)
    parser.add_argument("--probe-steps", type=int, default=0,
                        help="只跑这么多步测显存峰值就退出，不做验证/预测/落盘。"
                             "用于放大前的试点")
    parser.add_argument("--subset", type=int, default=0,
                        help="只用前 N 条训练样本（冒烟测试用）")
    parser.add_argument("--no-submission", action="store_true")
    return parser


def build_adalora_callback():
    """AdaLoRA 必须每步手动推进它的 rank 调度器，HF Trainer 不会替你做。

    peft 的 `update_and_allocate` docstring 写得很直白：
    「This should be called in every training step after `loss.backward()`
    and before `zero_grad()`」。它干两件事——从梯度算每个奇异值的重要性分数、
    按预算把不重要的裁掉。

    **不调用的后果不是报错，而是静默训不出来**（本项目实测）：
    rank 裁剪从未发生，而 `AdaLoraModel.forward` 里的正交正则项
    （`orth_reg_weight` 默认 0.5）却一直加在 loss 上——
    起始 loss 2.34 而不是 ln2≈0.693，模型把容量全用来压这一项，
    最后塌成全预测同一类，验证准确率 0.5094（等于瞎猜）。

    `on_pre_optimizer_step` 正好是「反向传播完、参数更新前」这个时点，
    梯度还在，符合 peft 的要求。fp16 下此时梯度仍带 GradScaler 的缩放因子，
    但所有梯度是同一个系数，重要性**排序**不受影响，裁剪结果一致。
    """
    from transformers import TrainerCallback

    class AdaLoraScheduleCallback(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            # PeftModel.base_model 才是 AdaLoraModel
            target = getattr(model, "base_model", model)
            update = getattr(target, "update_and_allocate", None)
            if update is not None:
                update(state.global_step)

    return AdaLoraScheduleCallback()


def predict_in_order(trainer, dataset):
    """按数据集**原始顺序**预测，返回 logits。

    这个函数存在的唯一理由是一句藏在 `Trainer._get_eval_sampler` 里的判断：

        if self.args.group_by_length:
            return LengthGroupedSampler(...)      # 不是 SequentialSampler！

    也就是说 `group_by_length` 这个开关**不只作用于训练，也作用于预测**。
    我们开它是为了减少 padding、压低显存峰值，但它会让 `predict()` 返回
    「按长度分组后」的顺序。把这样的概率和按文件原序排列的 id 配对，就是逐行错位。

    **后果极其隐蔽**：提交文件的行数、格式、概率分布看起来全都正常，
    只有分数等于随机（实测 ROC-AUC 0.5021，而同一个模型的验证集准确率是 0.9566）。
    验证集不受影响，因为 `compute_metrics` 拿到的 logits 和 labels 是同一个置换顺序，
    一一对应。所以「验证集很高 + 提交文件却是随机」正是这个 bug 的指纹。
    """
    trainer.args.group_by_length = False
    predictions = trainer.predict(dataset).predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    return predictions


def _resolve_accum(args: argparse.Namespace) -> int:
    if args.grad_accum > 0:
        return args.grad_accum
    return max(1, EFFECTIVE_BATCH // max(args.batch_size, 1))


def _load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(common.CORPUS_DIR / "labeledTrainData.tsv", header=0,
                        delimiter="\t", quoting=csv.QUOTE_NONE)
    test = pd.read_csv(common.CORPUS_DIR / "testData.tsv", header=0,
                       delimiter="\t", quoting=csv.QUOTE_NONE)
    return train, test


def _build_peft_config(args: argparse.Namespace, total_steps: int):
    """按方法造 PeftConfig。target_modules 留空，让 peft 用它对 DeBERTa 的默认映射
    （query_proj / value_proj）——demo 里注释掉的 `['q_proj','v_proj']` 是 LLaMA 的命名，
    在 DeBERTa 上会报「找不到目标模块」。"""
    from peft import (
        AdaLoraConfig,
        LoraConfig,
        PrefixTuningConfig,
        PromptEncoderConfig,
        TaskType,
    )

    # DeBERTa 的分类结构是 encoder → pooler.dense(1024×1024) → classifier(1024→2)，
    # 而 **pooler 和 classifier 在 from_pretrained 时都是随机初始化的**
    # （日志里那句 "newly initialized: classifier.*, pooler.dense.*"）。
    # peft 的 SEQ_CLS 只会自动把 classifier 放进 modules_to_save，pooler 既不训练也不保存：
    #   · 训练时它固定在一份随机权重上，LoRA 学着去配合那个随机投影；
    #   · 重新加载时 from_pretrained 生成**另一个**随机 pooler，学到的方向全部失效
    #     （实测验证集 0.9566 的模型，重载后测试集只有 0.4417、AUC 0.3116——反相关）。
    # 所以必须显式把 pooler 加进来。它同时也**该**被训练：
    # 让一个随机初始化的 1024×1024 投影全程冻结本来就没有道理。
    trainable_heads = ["pooler"]

    if args.method == "lora":
        return LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            modules_to_save=trainable_heads,
        )
    if args.method == "adalora":
        # peft 0.15 起 AdaLoraConfig 必须给 total_step：rank 的裁剪计划
        # （init warmup → 逐步裁剪 → final warmup）是按总步数排的，
        # 缺这个参数在 0.20 里直接 ValueError。
        return AdaLoraConfig(
            task_type=TaskType.SEQ_CLS,
            # 必须显式写死，不能像 LoRA 那样留空！
            # peft 给 AdaLoRA 的默认映射比 LoRA 宽得多——除了 query/value_proj
            # 还包括 key_proj 和**所有 FFN 的 dense**，实测挂载点从 48 处涨到 290 处、
            # 可训练参数从 157 万涨到 1422 万（9 倍）。那样比出来的不是
            # 「同样预算会不会分配」，而是「谁的预算大」，实验就没意义了。
            target_modules=["query_proj", "value_proj"],
            init_r=args.rank * 2,      # 从 2r 起，最终裁到平均 r
            target_r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            total_step=total_steps,
            tinit=max(1, total_steps // 10),
            tfinal=max(1, total_steps // 10),
            deltaT=10,
            bias="none",
            modules_to_save=trainable_heads,
        )
    if args.method == "prefix":
        return PrefixTuningConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=args.virtual_tokens,
        )   # prompt 类方法不支持 modules_to_save；roberta 的分类头本来就整个在
            # classifier.* 下（dense + out_proj），会被完整保存，不受这个坑影响
    if args.method == "ptuning":
        return PromptEncoderConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=args.virtual_tokens,
            encoder_hidden_size=128,
        )
    raise ValueError(f"未知方法 {args.method}")


def run(name: str, args: argparse.Namespace) -> dict | None:
    import datasets
    from peft import get_peft_model
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    device = common.setup(name, args.seed)
    mem_guard.cap_gpu(args.gpu_limit)
    accum = _resolve_accum(args)
    logging.info(
        "%s on %s：batch=%d × accum=%d（等效 %d）, max_length=%d, lr=%g, "
        "epochs=%s, checkpointing=%s, fp16=%s",
        METHOD_LABELS[args.method], args.model_id, args.batch_size, accum,
        args.batch_size * accum, args.max_length, args.lr, args.epochs,
        args.gradient_checkpointing, args.fp16)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    checkpoint_dir = common.MODELS_DIR / f"{name}_peft"

    train_frame, test_frame = _load_frames()
    # 与 hf_trainer.py / experiments/preprocess.py 完全一致的划分，保证跨阶段可比
    fit_frame, val_frame = train_test_split(
        train_frame, test_size=0.2, random_state=args.seed,
        stratify=train_frame["sentiment"])
    if args.subset:
        fit_frame = fit_frame.head(args.subset)
        val_frame = val_frame.head(max(args.subset // 4, 8))
        # 故意**不**截断测试集。冒烟测试要能验证「提交文件的概率和 id 有没有对上」——
        # 那是本阶段踩过最贵的一个坑（group_by_length 让 predict 按长度重排，
        # 提交文件格式全对但分数等于随机）。截断了就没法拿公开标签打分核对，
        # 这类 bug 就只能等跑完整套才发现。多花几分钟推理，换一个能真正兜住它的冒烟测试。
    logging.info("划分：训练 %d 条 / 验证 %d 条", len(fit_frame), len(val_frame))

    def to_dataset(frame: pd.DataFrame, labelled: bool = True) -> "datasets.Dataset":
        payload = {"text": frame["review"].tolist()}
        if labelled:
            payload["label"] = frame["sentiment"].tolist()
        return datasets.Dataset.from_dict(payload)

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_length)

    tokenized_train = to_dataset(fit_frame).map(tokenize, batched=True)
    tokenized_val = to_dataset(val_frame).map(tokenize, batched=True)
    tokenized_test = to_dataset(test_frame, labelled=False).map(tokenize, batched=True)

    steps_per_epoch = max(1, len(tokenized_train) // (args.batch_size * accum))
    total_steps = args.probe_steps or max(1, int(steps_per_epoch * args.epochs))

    mem_guard.reset_peak()
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, num_labels=2,
        # 默认 fp32 加载。`fp16=True` 的混合精度是在**计算**时转半精度，
        # activation 已经省了一半，权重仍需 fp32 的 master copy——
        # 直接用 fp16 权重会让 LoRA 参数也变成 fp16，而 GradScaler 明确拒绝
        # unscale fp16 梯度（ValueError: Attempting to unscale FP16 gradients）。
        # 只有底座大到 fp32 权重都装不下时才需要 --load-fp16，见下面的补救。
        torch_dtype=torch.float16 if args.load_fp16 else None,
    )
    mem_guard.snapshot("base_loaded")
    mem_guard.check("base_loaded", gpu_limit=args.gpu_limit, ram_limit=args.ram_limit)

    # peft 在 get_peft_model 里硬拦这个组合：
    # "PREFIX_TUNING does not work with gradient checkpointing."
    # 原因是前缀通过 past_key_values 注入，而 checkpoint 段反向重算时那份缓存
    # 已经不在图里了。这是方法与实现的真实约束，不是可以绕过的配置问题——
    # 自动关掉并如实记录，让对比表里 Prefix 的显存数字有个明确的脚注。
    if args.method == "prefix" and args.gradient_checkpointing:
        logging.warning("Prefix-Tuning 不支持 gradient checkpointing（peft 限制），"
                        "本次自动关闭——显存峰值会明显高于其他三种方法")
        args.gradient_checkpointing = False

    if args.gradient_checkpointing:
        # 必须在 get_peft_model 之前开，且要 enable_input_require_grads()：
        # 底座整个冻结时，第一层的输入没有 requires_grad，checkpoint 段会认为
        # 「这段不需要梯度」而不建图，反向传播直接报 "element 0 of tensors does not
        # require grad"。这是 LoRA + gradient checkpointing 最常见的坑。
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
            # reentrant 版本和冻结参数、和 Trainer 的 use_cache 处理都容易打架，
            # PyTorch 2.x 推荐非 reentrant 实现。
            "use_reentrant": False,
        })
        model.enable_input_require_grads()

    peft_config = _build_peft_config(args, total_steps)
    model = get_peft_model(model, peft_config)

    if args.load_fp16:
        # 底座留在 fp16，但把可训练的那 0.x%（LoRA 矩阵 + 分类头）转回 fp32，
        # 否则 GradScaler 报 "Attempting to unscale FP16 gradients"。
        # 这就是 peft 官方 `prepare_model_for_kbit_training` 做的同一件事，
        # 额外显存开销可以忽略（几百万参数）。
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.float()

    total_params, trainable_params = common.count_parameters(model)
    logging.info("参数量: 总计 %s, 可训练 %s（%.3f%%）", f"{total_params:,}",
                 f"{trainable_params:,}", 100 * trainable_params / total_params)
    mem_guard.snapshot("peft_wrapped")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        return {"accuracy": float((np.argmax(logits, axis=-1) == labels).mean())}

    training_args = TrainingArguments(
        output_dir=str(common.MODELS_DIR / f"{name}_peft_ckpt"),
        num_train_epochs=args.epochs,
        max_steps=args.probe_steps or -1,
        per_device_train_batch_size=args.batch_size,
        # 验证只做前向，没有梯度和 activation 缓存，batch 可以给大一些
        per_device_eval_batch_size=max(args.batch_size, 8),
        gradient_accumulation_steps=accum,
        learning_rate=args.lr,
        warmup_ratio=0.06,
        weight_decay=0.01,
        logging_dir=str(common.LOGS_DIR / f"{name}_peft"),
        logging_steps=50,
        eval_strategy="no" if args.probe_steps else "epoch",
        save_strategy="no",
        seed=args.seed,
        fp16=args.fp16,
        # 冻结底座后 dataloader 反而可能成为瓶颈，但 worker 会各持一份数据集副本，
        # 主机内存吃紧时得不偿失——保持 0，内存预算全留给模型。
        dataloader_num_workers=0,
        # 长度相近的样本放进同一个 batch，padding 大幅减少。
        # 变长 batch 下这是压低显存峰值最有效的一招（IMDB 长度方差很大）。
        group_by_length=True,
        report_to=[],
        label_names=["labels"],   # PeftModel 不暴露原始签名，不显式给会警告并漏算指标
        disable_tqdm=not sys.stderr.isatty(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[mem_guard.build_callback(gpu_limit=args.gpu_limit,
                                            ram_limit=args.ram_limit)],
    )
    if args.method == "adalora":
        trainer.add_callback(build_adalora_callback())

    started = time.time()
    trainer.train()
    elapsed = time.time() - started
    peak_gpu = mem_guard.gpu_peak_gb()
    mem_guard.snapshot("after_train")

    if getattr(trainer.state, "mem_guard_aborted", None):
        # 看门狗中止过：这一档的结果不可信，绝不能写进对比表冒充一次正常实验
        logging.error("训练被显存看门狗中止，不写结果：%s",
                      trainer.state.mem_guard_aborted)
        shutil.rmtree(common.MODELS_DIR / f"{name}_peft_ckpt", ignore_errors=True)
        return None

    if args.probe_steps:
        logging.info("探测完成：%d 步，显存峰值 %.2f GB，%.1fs",
                     args.probe_steps, peak_gpu, elapsed)
        shutil.rmtree(common.MODELS_DIR / f"{name}_peft_ckpt", ignore_errors=True)
        return {"probe": True, "peak_gpu_gb": round(peak_gpu, 2),
                "seconds": round(elapsed, 1), "batch_size": args.batch_size,
                "max_length": args.max_length, "model_id": args.model_id}

    evals = [entry for entry in trainer.state.log_history if "eval_accuracy" in entry]
    accuracy = float(evals[-1]["eval_accuracy"]) if evals else float(
        trainer.evaluate()["eval_accuracy"])
    logging.info("验证准确率 %.4f，训练 %.1fs，显存峰值 %.2f GB",
                 accuracy, elapsed, peak_gpu)

    # 只存 adapter（几 MB），不存底座——PEFT 的核心卖点之一
    trainer.model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    adapter_mb = sum(f.stat().st_size for f in checkpoint_dir.rglob("*")
                     if f.is_file()) / 1024 ** 2
    logging.info("已保存 adapter 到 %s（%.1f MB）", checkpoint_dir, adapter_mb)
    shutil.rmtree(common.MODELS_DIR / f"{name}_peft_ckpt", ignore_errors=True)

    history = []
    for entry in trainer.state.log_history:
        if "loss" in entry:
            logging.info("step %-6s epoch %.2f  train_loss %.4f  lr %.2e",
                         entry.get("step"), entry.get("epoch", 0.0), entry["loss"],
                         entry.get("learning_rate", 0.0))
        if "eval_accuracy" in entry:
            logging.info("epoch %.2f  val_loss %.4f  val_acc %.4f  %.1fs",
                         entry.get("epoch", 0.0), entry.get("eval_loss", float("nan")),
                         entry["eval_accuracy"], entry.get("eval_runtime", 0.0))
            history.append({
                "epoch": round(entry.get("epoch", 0.0), 4),
                "eval_loss": entry.get("eval_loss"),
                "eval_acc": entry["eval_accuracy"],
                "eval_seconds": entry.get("eval_runtime"),
            })
    if history:
        path = common.RESULTS_DIR / f"{name}_history.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

    if not args.no_submission:
        # 必须走 predict_in_order：`group_by_length` 会让 predict 按长度重排，
        # 和按文件原序的 id 配对就是逐行错位。详见该函数的 docstring。
        logits = predict_in_order(trainer, tokenized_test)
        # 竞赛指标是 ROC-AUC，交概率而非 0/1 硬标签（理由见 tools/score_test.py）
        probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=1)[:, 1]
        path = common.RESULTS_DIR / f"{name}_submission.csv"
        # 逐字节对齐 sampleSubmission.csv：标题行带引号，id 本身也带引号
        # （QUOTE_NONE 读入，所以 id 字符串里已含引号）。和 tools/score_test.py
        # 的 write_submission 保持完全一致的格式。
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write('"id","sentiment"\n')
            for identifier, probability in zip(test_frame["id"], probs.tolist()):
                handle.write(f"{identifier},{probability:.6f}\n")
        logging.info("已写出 %s（%d 行）", path.name, len(probs))

    summary = {
        "model": name,
        "method": METHOD_LABELS[args.method],
        "text_representation": f"{args.model_id} + {METHOD_LABELS[args.method]}（冻结底座）",
        "best_val_acc": round(accuracy, 4),
        "best_epoch": len(history) or None,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": accum,
        "effective_batch": args.batch_size * accum,
        "max_length": args.max_length,
        "learning_rate": args.lr,
        "gradient_checkpointing": args.gradient_checkpointing,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_pct": round(100 * trainable_params / total_params, 4),
        "adapter_mb": round(adapter_mb, 1),
        "peak_gpu_gb": round(peak_gpu, 2),
        "train_seconds": round(elapsed, 1),
        "seconds_per_epoch": round(elapsed / max(args.epochs, 1), 1),
        "device": str(device),
        "checkpoint": str(checkpoint_dir.relative_to(common.ROOT)),
    }
    path = common.RESULTS_DIR / f"{name}_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("已写出 %s", path.name)
    return summary
