"""路线①：继承 BertPreTrainedModel，在 forward 里实现 R-Drop。

R-Drop (https://arxiv.org/abs/2106.14448) 的做法：
  同一批输入过两次网络，因为 dropout 每次采样的 mask 不同，
  会得到两个不同的预测分布 p1、p2。除了各自的交叉熵之外，
  再加一项对称 KL 把两者拉近，逼模型对 dropout 噪声不敏感。

  L = 0.5 * (CE(p1, y) + CE(p2, y)) + alpha * 0.5 * (KL(p1||p2) + KL(p2||p1))

注意：模型必须真的开着 dropout，否则两次前向输出完全一样，KL 恒为 0，
      R-Drop 退化成"训练两倍 batch 的普通微调"。

用法:
    python imdb_bert_rdrop.py --alpha 1.0
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

REG_NAME = "rdrop"
from losses import rdrop_kl_loss


class BertForRDrop(BertPreTrainedModel):
    def __init__(self, config, alpha=1.0):
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

        self.post_init()

    def _head(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        pooled_output = self.dropout(outputs[1])
        return self.classifier(pooled_output), outputs

    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, labels=None, **kwargs):
        logits, outputs = self._head(input_ids, attention_mask, token_type_ids)

        loss = None
        if labels is not None and self.training:
            # 第二次前向：权重相同，但 dropout 重新采样
            logits2, _ = self._head(input_ids, attention_mask, token_type_ids)

            loss_fct = nn.CrossEntropyLoss()
            ce1 = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            ce2 = loss_fct(logits2.view(-1, self.num_labels), labels.view(-1))
            kl = rdrop_kl_loss(logits, logits2)
            loss = 0.5 * (ce1 + ce2) + self.alpha * kl
        elif labels is not None:
            # 评测时只走一次前向，dropout 已关闭，跑两次没有意义
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

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
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="KL 项权重，R-Drop 论文在 NLU 任务上取 1~5")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--limit", type=int, default=None,
                        help="只跑前 N 条做 smoke test")
    parser.add_argument("--output_dir", default=os.path.join(
        data.MODELS_DIR, "bert_base_rdrop"))
    parser.add_argument("--submission", default=os.path.join(
        data.SUBMISSION_DIR, "bert-base_full_rdrop.csv"))
    return parser.parse_args()


def main():
    logger = utils.setup_logging()
    args = parse_args()
    utils.set_seed(args.seed)

    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    train_ds, val_ds, test_ds, test_ids = data.build_datasets(
        tokenizer, data_dir=args.data_dir, max_length=args.max_length,
        seed=args.seed, limit=args.limit)

    model = BertForRDrop.from_pretrained(args.model_name, num_labels=2,
                                         alpha=args.alpha)

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
