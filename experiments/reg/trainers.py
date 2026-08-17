"""路线②：继承 Trainer，重写 compute_loss。

好处是完全不碰模型内部，所以任何 AutoModelForSequenceClassification
（ModernBERT / DeBERTa / …）套上 LoRA、套上 unsloth 之后都能直接用，
不需要为每种骨干网络重写一遍 forward。
"""
import logging
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import Trainer

from losses import SCLLoss, rdrop_kl_loss

logger = logging.getLogger(__name__)


def masked_mean_pool(hidden_states, attention_mask):
    """对 last_hidden_state 做 mask 平均池化，得到句向量。

    ModernBERT / DeBERTa 这类模型没有 BERT 的 pooler 层，
    所以不能像 `outputs[1]` 那样直接取，统一自己池化最稳。
    """
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


def find_encoder(model):
    """从（可能被 peft / unsloth 包过的）分类模型里找出骨干编码器。

    HF 的约定是 `base_model_prefix`：ModernBERT 是 "model"，DeBERTa 是
    "deberta"，BERT 是 "bert"。找到它才能挂 hook 拿最后一层隐藏状态。
    """
    m = model
    for _ in range(4):
        if hasattr(m, "get_base_model"):      # PeftModel
            m = m.get_base_model()
        elif hasattr(m, "module"):            # DDP / accelerate 包装
            m = m.module
        else:
            break
    prefix = getattr(m, "base_model_prefix", None)
    if prefix and hasattr(m, prefix):
        return getattr(m, prefix)
    raise RuntimeError(f"找不到 {type(m).__name__} 的骨干编码器，"
                       f"base_model_prefix={prefix}")


class RegularizedTrainer(Trainer):
    """在标准交叉熵之外，按需叠加 R-Drop 和 / 或 SCL。

    Args:
        reg: "none" | "rdrop" | "scl" | "both"
        rdrop_alpha: R-Drop 的 KL 项权重
        scl_alpha: SCL 项权重
        scl_temperature: SCL 温度
        safe_prediction_step: True 时绕过 unsloth 对 prediction_step 的
            monkey-patch，直接用 model(**inputs) 取 logits
    """

    def __init__(self, *args, reg="none", rdrop_alpha=1.0, scl_alpha=0.2,
                 scl_temperature=0.3, safe_prediction_step=False, **kwargs):
        super().__init__(*args, **kwargs)
        if reg not in ("none", "rdrop", "scl", "both"):
            raise ValueError(f"unknown reg: {reg}")
        self.reg = reg
        self.rdrop_alpha = rdrop_alpha
        self.scl_alpha = scl_alpha
        self.safe_prediction_step = safe_prediction_step
        self.use_rdrop = reg in ("rdrop", "both")
        self.use_scl = reg in ("scl", "both")
        self.scl_fct = SCLLoss(temperature=scl_temperature,
                               base_temperature=scl_temperature) if self.use_scl else None
        self.ce_fct = torch.nn.CrossEntropyLoss()
        self._hidden = {}
        self._hook_handle = None
        logger.info("RegularizedTrainer: reg=%s rdrop_alpha=%s scl_alpha=%s",
                    reg, rdrop_alpha, scl_alpha)

    def _register_hidden_hook(self, model):
        """只抓最后一层隐藏状态。

        为什么不用 output_hidden_states=True：那会把全部 N 层的输出都留在显存里，
        把 gradient checkpointing 省下来的显存又吃回去（ModernBERT-large 有 28 层，
        batch 16 / len 256 时直接 OOM）。hook 只留一份，显存差一个数量级。
        """
        if self._hook_handle is not None:
            return
        encoder = find_encoder(model)

        def hook(_module, _inputs, output):
            self._hidden["last"] = output[0] if isinstance(output, (tuple, list)) \
                else output.last_hidden_state

        self._hook_handle = encoder.register_forward_hook(hook)
        logger.info("registered hidden-state hook on %s", type(encoder).__name__)

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        # labels 自己接管，不交给模型内部的 loss，否则会重复计算
        inputs = dict(inputs)
        labels = inputs.pop("labels", None)
        if labels is None:
            labels = inputs.pop("label", None)

        if self.use_scl:
            self._register_hidden_hook(model)

        outputs = model(**inputs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]

        if labels is None:
            return (None, outputs) if return_outputs else None

        num_labels = logits.size(-1)
        loss = self.ce_fct(logits.view(-1, num_labels), labels.view(-1))

        if self.use_scl:
            features = masked_mean_pool(self._hidden["last"],
                                        inputs["attention_mask"])
            loss = loss + self.scl_alpha * self.scl_fct(features, labels)

        if self.use_rdrop and model.training:
            # 第二次前向：dropout 重新采样，得到另一组 logits
            outputs2 = model(**inputs)
            logits2 = outputs2["logits"] if isinstance(outputs2, dict) else outputs2[0]
            ce2 = self.ce_fct(logits2.view(-1, num_labels), labels.view(-1))
            kl = rdrop_kl_loss(logits, logits2)
            # CE 部分取两路平均，再加 KL 一致性项
            loss = 0.5 * (loss + ce2) + self.rdrop_alpha * kl

        self._hidden.clear()
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs: Dict[str, Any],
                        prediction_loss_only: bool,
                        ignore_keys: Optional[Tuple[str]] = None):
        if not self.safe_prediction_step:
            return super().prediction_step(model, inputs, prediction_loss_only,
                                           ignore_keys=ignore_keys)

        # unsloth 会给模型打上自己的 prediction_step，走标准路径可能拿不到 logits，
        # 这里直接调 forward 自己解包。
        inputs = self._prepare_inputs(inputs)
        inputs = dict(inputs)
        labels = inputs.pop("labels", inputs.pop("label", None))

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
            loss = None
            if labels is not None:
                loss = self.ce_fct(logits.view(-1, logits.size(-1)),
                                   labels.view(-1)).detach()

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits.detach().float(), labels)
