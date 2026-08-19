#!/usr/bin/env bash
# 第二阶段：真实 batch_size=16 的三组对照（grad_accum 自动 = 2，effective batch 仍是 32）。
set -euo pipefail
cd "$(dirname "$0")"
BS=16 "$(dirname "$0")/run_stage1.sh"
