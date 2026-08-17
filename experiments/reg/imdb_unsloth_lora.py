"""最终目标脚本：unsloth 加载骨干 + LoRA + 自定义 compute_loss（R-Drop / SCL）。

这就是老师说的 "用 unsloth 来调用你自己封装的，再挂 lora"：
    骨干模型 (ModernBERT / DeBERTa)
      -> unsloth FastModel 加载（省显存、开 gradient checkpointing）
      -> get_peft_model 挂 LoRA（只训练低秩增量）
      -> RegularizedTrainer 重写 compute_loss，叠加 R-Drop / SCL

四组对比实验:
    python imdb_unsloth_lora.py --reg none      # baseline: LoRA
    python imdb_unsloth_lora.py --reg rdrop     # + R-Drop
    python imdb_unsloth_lora.py --reg scl       # + SCL
    python imdb_unsloth_lora.py --reg both      # + 两者

显存不够时的顺序（老师给过的建议）：
    降 --batch_size -> 提 --grad_accum -> 开 --load_in_4bit -> 降 --max_length
"""
import os

# 需要走镜像时在 shell 里设 HF_ENDPOINT=https://hf-mirror.com；
# 注意 hf-mirror 对部分文件不返回 etag，huggingface_hub 会拒收，直连正常时别设。
# transformers 会尝试 import TF；本机装的是 Keras 3，不关掉会 import 失败
os.environ.setdefault("USE_TF", "0")
# 减少显存碎片，长短句混合时有用
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# unsloth 必须在 transformers 之前 import，它要 patch 一堆东西。
# 本机没装时自动退回 peft，代码路径其他部分完全一致。
try:
    import unsloth  # noqa: F401
    from unsloth import FastModel

    HAS_UNSLOTH = True
except ImportError:  # pragma: no cover
    FastModel = None
    HAS_UNSLOTH = False

import argparse
import json

import numpy as np
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, TrainingArguments)

import data
import utils
from trainers import RegularizedTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="answerdotai/ModernBERT-large",
                        help="也可以换成 microsoft/deberta-v3-large 等")
    parser.add_argument("--data_dir", default=data.DEFAULT_DATA_DIR)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_labels", type=int, default=2)

    parser.add_argument("--reg", default="none",
                        choices=["none", "rdrop", "scl", "both"])
    parser.add_argument("--rdrop_alpha", type=float, default=1.0)
    parser.add_argument("--scl_alpha", type=float, default=0.2)
    parser.add_argument("--scl_temperature", type=float, default=0.3)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="R-Drop 需要网络里有随机性，所以这里默认给 0.05，"
                             "不要照抄 unsloth 例子里的 0")
    parser.add_argument("--lora_target_modules", default="auto",
                        help="逗号分隔的模块名；auto = 按 model_type 查内置表")
    parser.add_argument("--load_in_4bit", action="store_true")

    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="LoRA 只训练少量参数，学习率要比全量微调高一个量级")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--limit", type=int, default=None,
                        help="只跑前 N 条做 smoke test，正式实验不要用")
    parser.add_argument("--tag_suffix", default="",
                        help="加到输出文件名后缀，用于区分不同后端/设定")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--submission", default=None)
    return parser.parse_args()


# peft 只给常见骨干内置了 target_modules，ModernBERT 这类新模型没有，
# 不显式指定会报 "Please specify `target_modules`"。
DEFAULT_TARGET_MODULES = {
    # ModernBERT: attn.Wqkv / attn.Wo / mlp.Wi / mlp.Wo
    "modernbert": ["Wqkv", "Wo", "Wi"],
    "deberta-v2": ["query_proj", "key_proj", "value_proj", "dense"],
    "bert": ["query", "key", "value", "dense"],
    "roberta": ["query", "key", "value", "dense"],
}


def resolve_target_modules(args, model_type, logger):
    if args.lora_target_modules and args.lora_target_modules != "auto":
        return [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    targets = DEFAULT_TARGET_MODULES.get(model_type)
    logger.info("lora target_modules for %s: %s", model_type, targets or "peft 默认")
    return targets


def load_model_and_tokenizer(args, logger):
    dtype = utils.torch_dtype()

    if HAS_UNSLOTH:
        logger.info("using unsloth FastModel")
        model, tokenizer = FastModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=args.max_length,
            dtype=dtype,
            load_in_4bit=args.load_in_4bit,
            auto_model=AutoModelForSequenceClassification,
            num_labels=args.num_labels,
        )
        model = FastModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=resolve_target_modules(
                args, model.config.model_type, logger),
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            # 不在 back propagation 中保存 activation，显存占用降到 10% 左右
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
            use_rslora=False,
            loftq_config=None,
            task_type="SEQ_CLS",
        )
        return model, tokenizer

    logger.warning("unsloth 未安装，退回 transformers + peft（显存占用会更高）")
    from peft import LoraConfig, TaskType, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # 权重按 fp32 加载，混合精度交给 Trainer 的 AMP。
    # 如果把权重直接读成 fp16，LoRA 参数也会是 fp16，GradScaler.unscale_ 会报
    # "Attempting to unscale FP16 gradients"。
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=args.num_labels, torch_dtype=torch.float32)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        target_modules=resolve_target_modules(
            args, model.config.model_type, logger),
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
    ))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return model, tokenizer


def main():
    logger = utils.setup_logging()
    args = parse_args()
    utils.set_seed(args.seed)

    tag = f"{args.model_name.split('/')[-1]}_lora_{args.reg}{args.tag_suffix}"
    output_dir = args.output_dir or os.path.join(data.MODELS_DIR, tag)
    submission = args.submission or os.path.join(data.SUBMISSION_DIR, f"{tag}.csv")

    if args.reg in ("rdrop", "both") and args.lora_dropout == 0:
        logger.warning("reg=%s 但 lora_dropout=0，两次前向输出相同，KL 项恒为 0！",
                       args.reg)

    model, tokenizer = load_model_and_tokenizer(args, logger)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("trainable params: %d / %d (%.4f%%)",
                trainable, total, 100 * trainable / total)

    train_ds, val_ds, test_ds, test_ids = data.build_datasets(
        tokenizer, data_dir=args.data_dir, max_length=args.max_length,
        seed=args.seed, limit=args.limit)

    use_bf16, use_fp16 = utils.pick_precision()
    logger.info("precision: bf16=%s fp16=%s", use_bf16, use_fp16)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.06,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        bf16=use_bf16,
        fp16=use_fp16,
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="no",
        seed=args.seed,
        label_names=["labels"],  # compute_loss 里自己 pop labels，必须显式声明
        report_to=[],
    )

    trainer = RegularizedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=utils.build_compute_metrics(),
        reg=args.reg,
        rdrop_alpha=args.rdrop_alpha,
        scl_alpha=args.scl_alpha,
        scl_temperature=args.scl_temperature,
        safe_prediction_step=HAS_UNSLOTH,
    )

    trainer.train()

    val_metrics = trainer.evaluate()
    logger.info("val metrics: %s", val_metrics)

    logits = trainer.predict(test_ds).predictions
    preds = np.argmax(logits, axis=-1).flatten()
    data.save_submission(test_ids, preds, submission)

    utils.save_metrics(submission, {"args": vars(args),
                                    "val_metrics": val_metrics})


if __name__ == "__main__":
    main()
