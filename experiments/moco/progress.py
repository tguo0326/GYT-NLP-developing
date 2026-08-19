"""进度追踪：解析 logs/run_stage1.out 与各组日志，打印当前状态与 ETA。

    python progress.py
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# 每组的 optimizer step 总数：20000 条 / effective batch 32 × 2 epoch
TOTAL_STEPS = 1250
# 探测得到的 s/step，用于给还没开始的组估时
SPEED = {"baseline": 3.27, "scl": 6.41, "scl_moco": 4.33}
ORDER = ["baseline", "scl", "scl_moco"]


def gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=10).stdout.strip()
        util, mem = [x.strip() for x in out.split(",")]
        return f"GPU {util}% / {int(mem) / 1024:.2f} GB"
    except Exception as exc:                                  # pragma: no cover
        return f"GPU 状态未知（{exc}）"


def main(stage_out=ROOT / "logs" / "run_stage1.out", batch_size=4, seed=42):
    text = stage_out.read_text(errors="ignore") if stage_out.exists() else ""
    started = re.findall(r"===== (\S+ \S+) start (\S+)", text)
    finished = {tag for _, tag in re.findall(r"===== (\S+ \S+) done (\S+)", text)}
    failed = re.findall(r"!!!!! (\S+) 失败", text)

    print(f"=== 进度 @ {datetime.now():%F %T} ===  {gpu()}")
    if not started:
        print("还没有任何组启动")
        return

    done_summaries = []
    for path in sorted((ROOT / "results").glob(f"*_bs{batch_size}_seed{seed}_summary.json")):
        s = json.loads(path.read_text())
        done_summaries.append(s)

    current_tag = started[-1][1]
    current_method = current_tag.split("_")[2] if current_tag.count("_") >= 2 else "?"
    if current_tag.startswith("scl_moco_scl_moco"):
        current_method = "scl_moco"
    elif current_tag.startswith("scl_moco_scl"):
        current_method = "scl"
    elif current_tag.startswith("scl_moco_baseline"):
        current_method = "baseline"

    # 当前组的训练进度。三个来源，按可靠性排序：
    #   1) logs/<tag>.log 里 LogToFileCallback 写的 "[trainer] step N/M"（最准）
    #   2) 对比统计日志里的 opt_step=N（scl / scl_moco 才有）
    #   3) 都没有时用「已耗时 / 探测得到的 s/step」估算（baseline 那组走这条）
    step, cur_last, source = 0, None, "估算"
    cur_log_path = ROOT / "logs" / f"{current_tag}.log"
    cur_text = cur_log_path.read_text(errors="ignore") if cur_log_path.exists() else ""
    trainer_lines = re.findall(r"\[trainer\] step (\d+)/\S+ (.*)", cur_text)
    if trainer_lines:
        step = int(trainer_lines[-1][0])
        cur_last = dict(re.findall(r"(\w+)=([-\d.e+]+)", trainer_lines[-1][1]))
        source = "trainer 日志"
    else:
        opt_steps = re.findall(r"opt_step=(\d+)", cur_text)
        if opt_steps:
            step, source = int(opt_steps[-1]), "对比统计日志"
        else:
            t0 = datetime.strptime(started[-1][0], "%Y-%m-%d %H:%M:%S")
            elapsed = (datetime.now() - t0).total_seconds() - 90   # 扣掉加载+tokenize
            step = max(0, int(elapsed / SPEED.get(current_method, 4.0)))
            source = "按耗时估算"

    for tag_time, tag in started:
        method = ("scl_moco" if "scl_moco" in tag else
                  "scl" if "_scl_" in tag else "baseline")
        done = tag in finished
        s = next((x for x in done_summaries if x["tag"] == tag), None)
        if done and s:
            t = s.get("test_metrics") or {}
            v = s.get("val_metrics_last_epoch") or {}
            print(f"[完成] {method:9s} {s['train_seconds'] / 60:5.1f} min  "
                  f"峰值 {s['peak_gpu_gb']:.2f} GB  "
                  f"val acc {(v.get('accuracy') or v.get('eval_accuracy'))}  test acc {t.get('accuracy')}  "
                  f"AUC {t.get('roc_auc')}")
        elif done:
            print(f"[完成] {method:9s}（summary 还没写出来？）")
        elif tag in failed:
            print(f"[失败] {method:9s} 见 logs/{tag}.log")
        else:
            pct = 100 * step / TOTAL_STEPS
            eta = (TOTAL_STEPS - step) * SPEED.get(method, 4.0)
            print(f"[进行中] {method:9s} step {step}/{TOTAL_STEPS}（{pct:.1f}%，"
                  f"{source}）"
                  f" epoch {cur_last.get('epoch') if cur_last else '?'}"
                  f" loss {cur_last.get('loss') if cur_last else '?'}"
                  f"  剩余训练约 {eta / 60:.0f} min"
                  f"（+推理约 12 min）→ 预计 "
                  f"{(datetime.now() + timedelta(seconds=eta + 720)):%H:%M}")

    remaining = [m for m in ORDER
                 if not any(m == ("scl_moco" if "scl_moco" in t else
                                  "scl" if "_scl_" in t else "baseline")
                            for _, t in started)]
    if remaining:
        eta = sum(TOTAL_STEPS * SPEED[m] + 720 for m in remaining)
        print(f"[排队中] {', '.join(remaining)}  预计还需 {eta / 3600:.1f} h")
    if step >= TOTAL_STEPS:
        print("（当前组训练已跑完，正在做验证 / 25000 条测试推理，约 12 min）")

    # 对比统计（只有 scl / scl_moco 有）
    for method in ("scl", "scl_moco"):
        tag = f"scl_moco_{method}_bs{batch_size}_seed{seed}"
        log = ROOT / "logs" / f"{tag}.log"
        if not log.exists():
            continue
        lines = [l for l in log.read_text(errors="ignore").splitlines()
                 if f"[{method}]" in l]
        if lines:
            print(f"  {method} 最新对比统计：{lines[-1].split('INFO ')[-1]}")


if __name__ == "__main__":
    main()
