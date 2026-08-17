"""路线①：继承 BertPreTrainedModel，在 forward 里实现 SCL（监督对比学习）。

SupCon (https://arxiv.org/abs/2004.11362)、NLP 微调版 (https://arxiv.org/abs/2011.01403)：
  交叉熵只管"分对类"，不管特征空间长什么样。SCL 额外要求：
  batch 内同标签的句向量互相靠近、不同标签的互相推远。

  L = CE(y_hat, y) + alpha * SCL(z, y),   z = L2Normalize(pooled_output)

两个容易踩的坑：
  1. 特征必须 L2 归一化，否则除以 temperature 之后 logits 爆掉；
  2. SCL 需要 batch 里同类样本成对出现，batch size 太小（比如 2）时
     正样本对经常为 0，这一项基本失效。真实 batch 建议 >= 16，
     显存不够就靠 gradient_accumulation_steps 撑等效 batch。

用法:
    python imdb_bert_scl.py --alpha 0.2 --temperature 0.3
"""
import os

# transformers 会尝试 import TF；本机装的是 Keras 3，不关掉会 import 失败
os.environ.setdefault("USE_TF", "0")
# 需要走镜像时在 shell 里设 HF_ENDPOINT=https://hf-mirror.com；
# 注意 hf-mirror 对部分文件不返回 etag，huggingface_hub 会拒收，直连正常时别设。

import argparse

import numpy as np
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from transformers import BertTokenizerFast, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

import data
import utils

REG_NAME = "scl"
from losses import SCLLoss


class BertForSCL(BertPreTrainedModel):
    def __init__(self, config, alpha=0.2, temperature=0.3):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.alpha = alpha

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.scl_fct = SCLLoss(temperature=temperature, base_temperature=temperature)

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        pooled_output = self.dropout(outputs[1])
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            ce_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # 对比项作用在句表示上，不是 logits 上
            scl_loss = self.scl_fct(pooled_output, labels)
            loss = ce_loss + self.alpha * scl_loss

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--data_dir", default=data.DEFAULT_DATA_DIR)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.2, help="SCL 项权重")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="温度，视觉侧常用 0.07，文本侧论文用 0.1~0.5")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_dir", default=os.path.join(
        data.MODELS_DIR, "bert_base_scl"))
    parser.add_argument("--submission", default=os.path.join(
        data.SUBMISSION_DIR, "bert-base_full_scl.csv"))
    return parser.parse_args()


def main():
    logger = utils.setup_logging()
    args = parse_args()
    utils.set_seed(args.seed)

    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    train_ds, val_ds, test_ds, test_ids = data.build_datasets(
        tokenizer, data_dir=args.data_dir, max_length=args.max_length,
        seed=args.seed, limit=args.limit)

    model = BertForSCL.from_pretrained(args.model_name, num_labels=2,
                                       alpha=args.alpha,
                                       temperature=args.temperature)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.06,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        bf16=utils.pick_precision()[0],
        fp16=utils.pick_precision()[1],
        logging_dir="./logs",
        logging_steps=100,
        save_strategy="no",
        eval_strategy="epoch",
        seed=args.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=utils.build_compute_metrics(),
    )

    trainer.train()
    val_metrics = trainer.evaluate()
    logger.info("val metrics: %s", val_metrics)

    logits = trainer.predict(test_ds).predictions
    preds = np.argmax(logits, axis=-1).flatten()
    data.save_submission(test_ids, preds, args.submission)
    utils.save_metrics(args.submission, {
        "args": vars(args),
        "reg": REG_NAME if args.alpha > 0 else "none",
        "val_metrics": val_metrics,
    })


if __name__ == "__main__":
    main()
