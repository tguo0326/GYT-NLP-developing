"""阶段三的单元测试——不需要 GPU、不联网、不下载任何权重。

这里抓的是三类**不报错但会毁掉实验**的问题：

1. 等效批大小算错。`--batch-size 8` 时累积步数必须是 4，
   否则和阶段二的 11 个模型口径不一致，对比表里的差异就分不清是方法还是 batch；
2. PeftConfig 造错。比如 AdaLoRA 少传 `total_step`（peft 0.15 起必填）、
   或者 prompt 类方法漏了 `task_type`——前者直接崩，后者会静默走错分支；
3. 看门狗失效。越线不抛异常的看门狗等于没有，而它是「绝不让显存爆炸」的唯一保障。

    pytest tests/test_peft.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import mem_guard  # noqa: E402
from core import peft_trainer  # noqa: E402


def make_args(**overrides):
    """造一份默认参数，再按需覆盖。走真实的 build_parser，
    这样默认值改了测试会跟着变——而不是在测试里抄一份会漂移的副本。"""
    args = peft_trainer.build_parser(method=overrides.pop("method", "lora")).parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


# ── 等效批大小 ────────────────────────────────────────────────

@pytest.mark.parametrize("batch_size,expected_accum", [
    (32, 1),    # 显存够，不需要累积
    (16, 2),
    (8, 4),
    (4, 8),
    (2, 16),    # xxlarge 那一档
    (1, 32),
])
def test_grad_accum_keeps_effective_batch_at_32(batch_size, expected_accum):
    args = make_args(batch_size=batch_size, grad_accum=0)
    accum = peft_trainer._resolve_accum(args)
    assert accum == expected_accum
    # 真正要守住的不是累积步数本身，而是这个乘积
    assert batch_size * accum == peft_trainer.EFFECTIVE_BATCH


def test_explicit_grad_accum_wins():
    """显式指定时不要被自动计算覆盖——调参时需要能手动压过默认行为。"""
    args = make_args(batch_size=8, grad_accum=7)
    assert peft_trainer._resolve_accum(args) == 7


def test_grad_accum_never_zero():
    """batch size 比等效批还大时，累积步数会算出 0，
    传给 TrainingArguments 会直接崩。必须兜到 1。"""
    args = make_args(batch_size=64, grad_accum=0)
    assert peft_trainer._resolve_accum(args) == 1


# ── PeftConfig ───────────────────────────────────────────────

def test_lora_config_targets_deberta_projections():
    """target_modules 留空时，peft 必须能从模型类型推出 DeBERTa 的命名。
    demo 里注释掉的 ['q_proj','v_proj'] 是 LLaMA 的命名，在 DeBERTa 上会
    报「找不到目标模块」——所以这里确认我们没有写死任何名字。"""
    config = peft_trainer._build_peft_config(make_args(method="lora"), total_steps=100)
    assert config.r == 16
    assert config.lora_alpha == 32
    assert config.target_modules is None       # 交给 peft 的默认映射
    assert config.task_type == "SEQ_CLS"


def test_adalora_config_has_total_step():
    """peft 0.15 起 AdaLoraConfig 必须传 total_step，否则 ValueError。
    而且裁剪计划的三段必须落在总步数之内。"""
    config = peft_trainer._build_peft_config(make_args(method="adalora"), total_steps=500)
    assert config.total_step == 500
    assert config.init_r > config.target_r     # 从 2r 裁到 r
    assert config.tinit + config.tfinal < config.total_step


def test_adalora_pins_the_same_modules_as_lora():
    """这是实测踩过的坑：`target_modules` 留空时，peft 给 AdaLoRA 的默认映射比
    LoRA 宽得多（多了 key_proj 和所有 FFN 的 dense），挂载点 290 处 vs 48 处、
    参数 1422 万 vs 157 万。那样比出来的不是「同样预算会不会分配」而是
    「谁的预算大」——实验直接失效，而且**不报任何错**。

    所以 AdaLoRA 必须显式写死，且必须和 LoRA 完全一致。"""
    lora = peft_trainer._build_peft_config(make_args(method="lora"), total_steps=100)
    adalora = peft_trainer._build_peft_config(make_args(method="adalora"), total_steps=100)

    assert adalora.target_modules is not None, "AdaLoRA 不能留空，默认映射比 LoRA 宽"
    # LoRA 留空是有意的（用 peft 的默认映射），这里钉住那份默认值到底是什么
    assert set(adalora.target_modules) == {"query_proj", "value_proj"}


def test_adalora_callback_calls_update_and_allocate():
    """第二个静默坑：`update_and_allocate()` 必须每步手动调，HF Trainer 不会调。
    不调不报错，但 rank 裁剪从未发生、正交正则一直白压 loss，
    实测准确率 0.5094（二分类瞎猜）。这里用一个假 model 确认回调真的调了它。"""
    class FakeAdaLoraModel:
        def __init__(self):
            self.calls = []

        def update_and_allocate(self, step):
            self.calls.append(step)

    class FakePeftModel:
        def __init__(self):
            self.base_model = FakeAdaLoraModel()

    class FakeState:
        global_step = 42

    model = FakePeftModel()
    callback = peft_trainer.build_adalora_callback()
    callback.on_pre_optimizer_step(None, FakeState(), None, model=model)
    assert model.base_model.calls == [42]


def test_adalora_callback_is_harmless_on_plain_models():
    """回调对没有 update_and_allocate 的模型必须静默跳过，而不是 AttributeError。"""
    class Plain:
        pass

    class FakeState:
        global_step = 1

    callback = peft_trainer.build_adalora_callback()
    callback.on_pre_optimizer_step(None, FakeState(), None, model=Plain())


def test_adalora_schedule_survives_tiny_total_steps():
    """探测模式只跑 8 步，tinit/tfinal 会被算成 0——peft 对 0 是否合法不保证，
    所以代码里兜了 max(1, ...)。这里确认那个兜底还在。"""
    config = peft_trainer._build_peft_config(make_args(method="adalora"), total_steps=8)
    assert config.tinit >= 1
    assert config.tfinal >= 1


@pytest.mark.parametrize("method", ["prefix", "ptuning"])
def test_prompt_configs_carry_task_type_and_tokens(method):
    """prompt 类方法漏了 task_type 会静默走成通用分支，分类头不被保存。"""
    config = peft_trainer._build_peft_config(make_args(method=method), total_steps=100)
    assert config.task_type == "SEQ_CLS"
    assert config.num_virtual_tokens == 20


def test_unknown_method_raises():
    args = make_args(method="lora")
    args.method = "nope"          # 绕过 argparse 的 choices 校验，直接测函数本身
    with pytest.raises(ValueError):
        peft_trainer._build_peft_config(args, total_steps=100)


def test_every_config_saves_the_pooler_on_deberta():
    """DeBERTa 的 pooler.dense(1024×1024) 在 from_pretrained 时是**随机初始化**的，
    而 peft 的 SEQ_CLS 只自动保存 classifier。pooler 既不训练也不保存的话：

      · 训练时 LoRA 学着去配合那一份特定的随机投影；
      · 重新加载时 from_pretrained 生成另一个随机 pooler，学到的方向全部失效。

    实测：验证集 0.9566 的模型，重载后测试集 0.4417 / AUC 0.3116（反相关），
    **完全不报错**。所以 LoRA 系必须显式把 pooler 放进 modules_to_save。"""
    for method in ("lora", "adalora"):
        config = peft_trainer._build_peft_config(make_args(method=method), total_steps=100)
        assert config.modules_to_save is not None, f"{method} 没保存 pooler"
        assert "pooler" in config.modules_to_save, f"{method} 没保存 pooler"


def test_predict_in_order_disables_group_by_length():
    """最贵的一个坑：`Trainer._get_eval_sampler` 里 `if self.args.group_by_length`
    会返回 `LengthGroupedSampler` 而不是 `SequentialSampler`——这个开关
    **也作用于预测**。开着它 predict 出来的概率是按长度分组的顺序，
    和按文件原序的 id 配对就是逐行错位。

    后果：提交文件行数、格式、概率分布全部正常，分数却等于随机
    （实测 ROC-AUC 0.5021，而同一模型验证集 0.9566）。
    """
    class FakeArgs:
        group_by_length = True

    class FakeTrainer:
        def __init__(self):
            self.args = FakeArgs()
            self.saw_group_by_length = None

        def predict(self, dataset):
            # 记录 predict 被调用**那一刻**的开关状态——这才是真正要钉住的东西
            self.saw_group_by_length = self.args.group_by_length

            class Output:
                predictions = [[0.1, 0.9], [0.8, 0.2]]

            return Output()

    trainer = FakeTrainer()
    logits = peft_trainer.predict_in_order(trainer, dataset=None)
    assert trainer.saw_group_by_length is False, "predict 时 group_by_length 还开着！"
    assert logits == [[0.1, 0.9], [0.8, 0.2]]


def test_predict_in_order_unwraps_tuple_predictions():
    """有些 PEFT 方法的 predictions 是 tuple（logits, 其他）。取错会把
    整个 tuple 送进 softmax。"""
    class FakeArgs:
        group_by_length = True

    class FakeTrainer:
        args = FakeArgs()

        def predict(self, dataset):
            class Output:
                predictions = ([[1.0, 2.0]], "别的东西")

            return Output()

    assert peft_trainer.predict_in_order(FakeTrainer(), None) == [[1.0, 2.0]]


# ── 看门狗 ───────────────────────────────────────────────────

def test_guard_raises_when_ram_over_limit():
    """把上限压到 0 就必然越线。不抛异常的看门狗等于没有。"""
    with pytest.raises(mem_guard.MemoryBudgetExceeded):
        mem_guard.check("test", gpu_limit=1e9, ram_limit=0.0)


def test_guard_passes_under_generous_limits():
    mem_guard.check("test", gpu_limit=1e9, ram_limit=1e9)


def test_guard_error_message_names_the_stage():
    """报错信息里必须带阶段名，否则探测时分不清是加载崩的还是训练崩的。"""
    with pytest.raises(mem_guard.MemoryBudgetExceeded, match="load_base"):
        mem_guard.check("load_base", gpu_limit=1e9, ram_limit=0.0)


def test_ram_reading_is_positive():
    assert mem_guard.ram_gb() > 0


def test_reset_peak_is_safe_without_cuda():
    """CPU-only 环境（比如 CI）里这些函数不能崩。"""
    mem_guard.reset_peak()
    assert mem_guard.gpu_peak_gb() >= 0.0
    mem_guard.cap_gpu(13.5)      # 无 GPU 时应当直接 return


# ── 方法清单的一致性 ─────────────────────────────────────────

def test_every_method_has_a_label():
    assert set(peft_trainer.METHODS) == set(peft_trainer.METHOD_LABELS)


def test_collect_results_lists_every_peft_run():
    """对比表的模型清单里必须包含全部四个 PEFT 结果，
    否则跑完了却不出现在表里——最容易被忽略的一种静默失败。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import collect_results

    listed = {name for name, *_ in collect_results.MODELS}
    assert collect_results.PEFT <= listed


def test_score_test_knows_every_peft_base():
    """score_test 必须记得每个 adapter 当初挂在哪个底座上。adapter 里不含底座权重，
    记错了会加载出一个没微调过的模型，分数看起来只是「差一点」，很难发现。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import collect_results
    import score_test

    assert set(score_test.PEFT_MODELS) == collect_results.PEFT
    # Prefix 的底座必须不是 DeBERTa——DeBERTa 没有 KV cache，跑不了
    assert "deberta" not in score_test.PEFT_MODELS["deberta_prefix"][0]
