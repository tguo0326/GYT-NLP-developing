"""显存 / 内存看门狗：让「跑不动」变成一秒内的报错，而不是把机器拖死。

为什么需要它。PEFT 这一阶段要在单卡 Tesla T4（15 GB）上挂 4 亿甚至 15 亿参数的
底座，正常做法是「试一档、爆了退一档」。但 PyTorch 默认的行为对试错很不友好：

1. 缓存分配器会一路把显存吃到物理上限才抛 OOM，此时同卡上的其他进程一起完蛋；
2. 真正的峰值往往出现在训练开始后几十步，一次失败要等几分钟；
3. 更糟的是主机内存——`from_pretrained` 加载 15 亿参数的 fp32 权重要 6 GB，
   加上 safetensors 的临时拷贝可能翻倍，30 GB 的机器一旦触发 swap 就是整机卡死，
   连 Ctrl-C 都按不进去，而 CUDA 的 OOM 至少还能捕获。

所以这里做三件事：

    cap_gpu()      给本进程的显存设硬上限，超了立刻抛 CUDA OOM 而不是吃满整张卡
    MemoryGuard    Trainer 回调，每步查显存和主机内存，越线主动中止训练
    peak_report()  读取峰值，写进 summary，供对比表和文档引用

三者是递进的保险：cap_gpu 保护同机的其他进程，MemoryGuard 保护主机内存
（这块 PyTorch 管不到），peak_report 把「到底用了多少」变成可记录的数字。
"""

from __future__ import annotations

import gc
import logging

import psutil
import torch

# 默认上限。T4 是 15,360 MiB，留约 1.5 GB 给 CUDA context、cuDNN workspace 和
# 显存碎片——实测把上限顶到 14.5 GB 以上时，失败方式会从「干净的 OOM 异常」
# 退化成驱动层报错，进程 kill 不掉。
GPU_LIMIT_GB = 13.5
# 主机内存上限。机器 30 GB，其中约 26 GB 可用（其余是 buff/cache）。
# 22 GB 的线留出足够余量，保证绝不进 swap。
RAM_LIMIT_GB = 22.0

_BYTES_PER_GB = 1024 ** 3


class MemoryBudgetExceeded(RuntimeError):
    """越线时抛出。用独立类型，方便探测脚本区分「显存不够」和「代码有 bug」。"""


def total_gpu_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / _BYTES_PER_GB


def cap_gpu(limit_gb: float = GPU_LIMIT_GB) -> None:
    """把本进程可用显存限制在 limit_gb 以内。

    `set_per_process_memory_fraction` 作用在缓存分配器上：超过比例的申请直接抛
    `torch.cuda.OutOfMemoryError`，是个可以 try/except 的普通异常。
    没有这行的话，分配器会一路吃到 15 GB 物理上限——同卡的其他进程会被连带搞死。
    """
    if not torch.cuda.is_available():
        return
    total = total_gpu_gb()
    fraction = min(limit_gb / total, 1.0)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    logging.info("显存上限 %.1f GB / %.1f GB（fraction=%.3f）", limit_gb, total, fraction)


def gpu_peak_gb() -> float:
    """本进程显存峰值。用 reserved 而非 allocated：前者才是从驱动实际拿走的量，
    也就是别的进程看不到的那部分。"""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_reserved() / _BYTES_PER_GB


def ram_gb() -> float:
    """当前进程常驻内存（RSS）。"""
    return psutil.Process().memory_info().rss / _BYTES_PER_GB


def reset_peak() -> None:
    """清空峰值统计并回收缓存。每档探测之前调用，否则读到的是上一档的峰值。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def check(stage: str, *, gpu_limit: float = GPU_LIMIT_GB,
          ram_limit: float = RAM_LIMIT_GB) -> None:
    """越线就抛 MemoryBudgetExceeded。stage 只用于报错信息里定位阶段。"""
    peak = gpu_peak_gb()
    rss = ram_gb()
    if peak > gpu_limit:
        raise MemoryBudgetExceeded(
            f"{stage}: 显存峰值 {peak:.2f} GB 超过上限 {gpu_limit:.1f} GB")
    if rss > ram_limit:
        raise MemoryBudgetExceeded(
            f"{stage}: 主机内存 {rss:.2f} GB 超过上限 {ram_limit:.1f} GB")


def snapshot(stage: str) -> None:
    logging.info("[mem] %-14s 显存峰值 %.2f GB  主机内存 %.2f GB",
                 stage, gpu_peak_gb(), ram_gb())


def build_callback(*, gpu_limit: float = GPU_LIMIT_GB,
                   ram_limit: float = RAM_LIMIT_GB, every: int = 10):
    """返回一个 Trainer 回调，每 `every` 步检查一次预算。

    延迟 import transformers：`mem_guard` 也被探测脚本单独使用，
    那里不需要把整个 transformers 拖进来。

    检查间隔不设成 1 是因为 `memory_reserved()` 会同步 CUDA 流，每步都查会拖慢训练；
    每 10 步足够——显存峰值在前若干步就稳定了，之后基本是平的。
    """
    from transformers import TrainerCallback

    class MemoryGuard(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            snapshot("train_begin")
            check("train_begin", gpu_limit=gpu_limit, ram_limit=ram_limit)

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % every:
                return
            try:
                check(f"step {state.global_step}", gpu_limit=gpu_limit,
                      ram_limit=ram_limit)
            except MemoryBudgetExceeded as exc:
                # 让 Trainer 自己收尾退出循环，比在回调里抛异常干净：
                # 抛异常会留下半个 checkpoint 目录和没关掉的 dataloader worker。
                logging.error("%s —— 主动中止训练", exc)
                control.should_training_stop = True
                state.mem_guard_aborted = str(exc)

        def on_train_end(self, args, state, control, **kwargs):
            snapshot("train_end")

    return MemoryGuard()
