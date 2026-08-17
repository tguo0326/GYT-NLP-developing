"""日志、评测指标等公共工具。"""
import logging
import os
import random
import sys

import numpy as np
import torch


def setup_logging():
    program = os.path.basename(sys.argv[0])
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger = logging.getLogger(program)
    logger.info("running %s", " ".join(sys.argv))
    return logger


def set_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_metrics(submission_path, payload):
    """把超参和验证指标写到 submission csv 旁边，供 score_local.py 汇总。"""
    import json
    path = submission_path.replace(".csv", "_metrics.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logging.getLogger(__name__).info("metrics saved to %s", path)


def pick_precision():
    """返回 (use_bf16, use_fp16)。

    不要直接用 torch.cuda.is_bf16_supported()：新版 torch 在 T4（sm_75）上也返回
    True，但那是软件模拟的 bf16，比 fp16 慢好几倍。只有 Ampere 及以后
    （compute capability >= 8.0）才有原生 bf16。
    """
    if not torch.cuda.is_available():
        return False, False
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        return True, False
    return False, True


def torch_dtype():
    use_bf16, use_fp16 = pick_precision()
    if use_bf16:
        return torch.bfloat16
    if use_fp16:
        return torch.float16
    return torch.float32


def build_compute_metrics():
    """accuracy。注意新版 datasets 已删掉 load_metric，统一走 evaluate。"""
    import evaluate
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    return compute_metrics
