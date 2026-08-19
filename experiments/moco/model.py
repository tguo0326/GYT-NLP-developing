"""建模：DeBERTa-v3-large（冻结）+ LoRA + projection head，可选第二套动量 adapter。

LoRA 配置逐项对齐 0.9633 那次实验（`NLP/core/peft_trainer.py::_build_peft_config`）：
r=16 / alpha=32 / dropout=0.05 / bias=none / `target_modules` 留空走 peft 对 DeBERTa
的默认映射（query_proj、value_proj）/ `modules_to_save` 必须含 `pooler`。

`pooler` 那一条是原项目踩过的坑，这里照抄结论：DeBERTa 的 `pooler.dense`(1024×1024)
和 `classifier` 在 `from_pretrained` 时**都是随机初始化的**，peft 的 SEQ_CLS 只会自动
把 `classifier` 放进 `modules_to_save`。不显式加上 pooler，它就会全程冻结在一份随机
投影上，而且重新加载时会换成另一份随机权重（实测 val 0.9566 的模型重载后 test 只有
0.4417、AUC 0.3116，反相关）。

本文件相对原项目新增的只有一样：`proj_head`（对比学习用的投影头）。
它同样走 `modules_to_save`，于是每个 adapter 各有一份，切 adapter 时自动跟着切。
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from moco import (KEY_ADAPTER, QUERY_ADAPTER, FeatureQueue, MomentumBranch,
                  ProjectionHead, masked_mean_pool)

logger = logging.getLogger(__name__)

METHODS = ("baseline", "scl", "scl_moco")


def find_backbone(model):
    """从被 peft 包过的分类模型里找出骨干（DeBERTa 是 `deberta`）。"""
    inner = model
    for _ in range(4):
        if hasattr(inner, "get_base_model"):
            inner = inner.get_base_model()
        elif hasattr(inner, "module"):
            inner = inner.module
        else:
            break
    prefix = getattr(inner, "base_model_prefix", None)
    if prefix and hasattr(inner, prefix):
        return getattr(inner, prefix)
    raise RuntimeError(f"找不到 {type(inner).__name__} 的骨干，base_model_prefix={prefix}")


def find_module_by_suffix(model, suffix: str):
    for name, module in model.named_modules():
        if name.split(".")[-1] == suffix:
            return module
    raise RuntimeError(f"找不到名为 {suffix} 的模块")


class ModelBundle:
    """把「模型 + 取特征的方式 + 动量分支 + 队列」打包在一起。

    取特征的路径：
        backbone 的 last_hidden_state（forward hook 抓，不用 output_hidden_states=True，
        那会把 24 层全留在显存里，把 gradient checkpointing 省下的又吃回去）
        → masked_mean_pool → proj_head → L2 normalize
    分类路径完全不动，仍然是 peft 的 pooler → classifier → CE。
    """

    def __init__(self, model, tokenizer, method: str, proj_dim: int,
                 queue_size: int = 0):
        self.model = model
        self.tokenizer = tokenizer
        self.method = method
        self.hidden_store = {}
        self._hook = None
        self.proj_module = None
        self.momentum = None
        self.queue = None

        if method != "baseline":
            backbone = find_backbone(model)
            self._hook = backbone.register_forward_hook(self._capture)
            self.proj_module = find_module_by_suffix(model, "proj_head")
            logger.info("已在 %s 上挂 hidden-state hook", type(backbone).__name__)

        if method == "scl_moco":
            self.queue = FeatureQueue(queue_size, proj_dim).to(
                next(model.parameters()).device)
            self.momentum = MomentumBranch(model, self.features_from_last_forward_inputs)
            self.momentum.hard_sync()      # 初始化时 key 与 query 完全一致
            self.momentum.freeze_key()

    # ---- hook ----
    def _capture(self, _module, _inputs, output):
        self.hidden_store["last"] = (output[0] if isinstance(output, (tuple, list))
                                     else output.last_hidden_state)

    # ---- 特征 ----
    def project(self, attention_mask):
        """用上一次前向留在 hook 里的 hidden state 算投影特征（未归一化）。"""
        pooled = masked_mean_pool(self.hidden_store["last"], attention_mask)
        return self.proj_module(pooled)

    def features_from_last_forward_inputs(self, inputs):
        """给动量分支用：自己跑一次前向再取特征（inputs 不含 labels）。"""
        self.model(**inputs)
        return self.project(inputs["attention_mask"])

    def normalized_query_features(self, attention_mask):
        return F.normalize(self.project(attention_mask).float(), dim=-1)

    def close(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None


def build_lora_config(args, include_proj: bool):
    modules_to_save = ["pooler"] + (["proj_head"] if include_proj else [])
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        modules_to_save=modules_to_save,
    )


def build(args) -> ModelBundle:
    """按 args.method 建好模型。返回 ModelBundle。"""
    if args.method not in METHODS:
        raise ValueError(f"未知 method: {args.method}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    base = AutoModelForSequenceClassification.from_pretrained(args.model_id,
                                                             num_labels=args.num_labels)
    hidden_size = base.config.hidden_size
    # 三组实验都建同一个 proj_head（哪怕 baseline 用不到），
    # 这样模型构造消耗的随机数序列一致，底座/pooler/classifier 的初始化在三组间完全相同。
    # baseline 不把它放进 modules_to_save，于是它保持冻结、不进 optimizer、不参与训练。
    base.proj_head = ProjectionHead(hidden_size, args.proj_dim)

    if args.gradient_checkpointing:
        # 必须在 get_peft_model 之前开，并且要 enable_input_require_grads()：
        # 底座整个冻结时第一层输入没有 requires_grad，checkpoint 段不会建图，
        # 反向会报 "element 0 of tensors does not require grad"。
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()

    include_proj = args.method != "baseline"
    model = get_peft_model(base, build_lora_config(args, include_proj),
                           adapter_name=QUERY_ADAPTER)

    if args.method == "scl_moco":
        # 第二套 adapter：共享同一个冻结底座，只多一份 LoRA + pooler + proj_head + classifier
        model.add_adapter(KEY_ADAPTER, build_lora_config(args, include_proj))
        model.set_adapter(QUERY_ADAPTER)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("参数量：总计 %s，可训练 %s（%.4f%%）",
                f"{total:,}", f"{trainable:,}", 100 * trainable / total)

    bundle = ModelBundle(model, tokenizer, args.method, args.proj_dim,
                         queue_size=args.queue_size)
    return bundle


def save_moco_state(bundle: ModelBundle, path: str) -> None:
    """保存队列、指针、有效长度与动量参数（以及 query 侧可训练参数）。

    原项目的 `save_strategy="no"`，checkpoint 是训练结束后手动存的；
    这里额外把 MoCo 的状态一起存，保证「保存并重新加载后队列/指针/动量参数能恢复」。
    """
    state = {
        "method": bundle.method,
        "trainable": {n: p.detach().cpu().clone()
                      for n, p in bundle.model.named_parameters() if p.requires_grad},
    }
    if bundle.queue is not None:
        state["queue"] = {k: v.detach().cpu().clone()
                          for k, v in bundle.queue.state_dict().items()}
    if bundle.momentum is not None:
        state["key"] = [k.detach().cpu().clone() for k in bundle.momentum.key_params()]
    torch.save(state, path)
    logger.info("已保存 MoCo 状态到 %s", path)


def load_moco_state(bundle: ModelBundle, path: str) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    named = dict(bundle.model.named_parameters())
    with torch.no_grad():
        for name, tensor in state["trainable"].items():
            named[name].data.copy_(tensor.to(named[name].device))
        if bundle.queue is not None and "queue" in state:
            bundle.queue.load_state_dict(
                {k: v.to(bundle.queue.feats.device) for k, v in state["queue"].items()})
        if bundle.momentum is not None and "key" in state:
            for param, tensor in zip(bundle.momentum.key_params(), state["key"]):
                param.data.copy_(tensor.to(param.device))
    logger.info("已从 %s 恢复 MoCo 状态", path)
    return state
