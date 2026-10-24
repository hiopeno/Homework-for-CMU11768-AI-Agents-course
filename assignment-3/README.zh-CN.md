# scottyrl

<p align="center">
  <img
    src="assets/scottyrl-logo.webp"
    alt="ScottyRL 吉祥物：戴着红色格纹头巾、跃起接飞盘的机器人苏格兰梗"
    width="240"
  >
</p>

一个用于通过强化学习训练 LLM agent 的最小化框架，
为 **11-768（AI Agents）第 3 次作业**构建。

作者：Apurva Gandhi、Aditya Soni 和 Demi Wang；感谢 11-768 课程团队（授课教师 Daniel Fried 和 Graham Neubig，以及助教 Saujas Vaduguru 和 Andy Liu）的反馈。

**作业说明位于 [ASSIGNMENT.zh-CN.md](ASSIGNMENT.zh-CN.md)：**

## 环境设置

使用 Python 3.12 和 [uv](https://docs.astral.sh/uv/)（项目使用 3.12 测试；`pyproject.toml` 接受 3.11 或更高版本）：

```bash
uv python install 3.12
uv sync                        # 安装编写代码和运行测试所需的全部依赖
```

所有命令都应从仓库根目录通过 `uv run` 执行。公开测试、环境、`MockPolicy` rollout 以及完整数据流水线都可以离线运行。训练和评估需要兼容 Tinker 的后端：本作业中，你需要将 SkyRL server 部署到 Modal，并通过 `TINKER_BASE_URL` 让脚本指向它（参见 `skyrl_tinker_modal/README.zh-CN.md`）。wandb 是可选的（导出 `WANDB_API_KEY` 后使用 `--set train.wandb_project=<project>`）；`runs/` 下的 jsonl 日志始终会写入，并以此作为评分依据。

## 仓库导览

```
scottyrl/
├── ASSIGNMENT.md           作业说明：各部分、背景、评分和提示
├── README.md               本文件
├── pyproject.toml          依赖（tinker 固定为 0.24.0）
├── scottyrl/
│   ├── types.py            Task / LLMCall / Trajectory / TokenSequence；请先阅读
│   ├── config.py            ModelConfig / TrainConfig、YAML 加载器、--set 覆盖和运行名称保护
│   ├── policy.py            TinkerPolicy（真实后端）、CompletionsServerPolicy（vLLM/SGLang，已导出的 adapter）以及 MockPolicy（测试，无需密钥）
│   ├── chat_templates/
│   │   └── qwen3_5.jinja   Qwen3.5 chat template，以及 enable_thinking / preserve_thinking_history 开关
│   ├── agents/
│   │   ├── base.py         Agent 接口（异步 reset / step / close）和 AgentAction
│   │   ├── xml_tool_call.py  XMLToolCallAgent：每轮一个 <tool>...</tool> 调用，并在 token 空间中维护 prompt
│   │   └── __init__.py      agent 注册表（xml_toolcall）
│   ├── envs/
│   │   ├── base.py          异步 Env 接口；步数预算由环境负责
│   │   ├── number_search.py  Number Search：根据“过低 / 过高”反馈猜隐藏整数
│   │   ├── textcraft/
│   │   │   ├── env.py       TextCraft-Synth：沿配方 DAG 制作目标物品
│   │   │   └── recipe_book.py  加载随项目提供的配方书（data/textcraft_recipes.json）
│   │   └── __init__.py      环境注册表（number_search、textcraft）
│   ├── rollout.py           一个 episode：agent 与环境的交互，以及 logger 读取的轨迹缓冲区
│   ├── evaluate.py          在留出任务上计算 success rate / pass@k
│   ├── metrics.py           MetricsLogger（jsonl、轨迹、wandb）、pass_at_k、sanity_warnings
│   ├── advantages.py        ★ REINFORCE / GRPO 优势计算
│   ├── datagen.py           ★ trajectory -> 拼接且带 mask 的 token sequence
│   ├── tinker_adapter.py    ★ 只需实现 submit_update；TinkerSession 的其余部分已提供
│   └── train/
│       ├── common.py        ★ group rollout + batch 组装（两个 trainer 共用）
│       ├── sync.py           ★ 同步 trainer（SyncTrainer）
│       └── async_grpo.py     ★ 异步生产者—消费者 trainer（AsyncGRPOTrainer）
├── configs/
│   ├── p2a_number_search_reinforce.yaml  每个必需运行一个 YAML：p2a、p2b、p3、
│   ├── ...                                p41、p43_*_iw、p43_*_noiw、p5
│   └── required_runs.yaml                 run_name -> 七个必需运行的配置
├── scripts/
│   ├── train_sync.py          同步运行入口（p2a、p2b、p41）
│   ├── train_async.py         异步运行入口（p3、p43_*、p5）
│   ├── evaluate_checkpoint.py 在验证集上评估 checkpoint、已部署的 adapter 或 base model
│   ├── export_checkpoint.py   将第 5 部分的 adapter 导出到公开 Hugging Face 仓库
│   └── package_submission.py  检查必需运行并写入 submission.zip
├── data/                      随项目提供的任务划分和配方书；来源见 data/README.zh-CN.md
├── tests/                     公开测试集：test_envs.py、test_interfaces.py、test_contracts.py
├── assets/                    logo
└── skyrl_tinker_modal/        在 Modal 上部署的 SkyRL Tinker server，每类运行一个启动脚本
```

**★ 标记六个需要你实现代码的文件。** 在 `tinker_adapter.py` 中，只有 `TinkerSession.submit_update` 属于你的实现。每个 stub 都标记为 `TODO(student)`，并会抛出 `NotImplementedError`；请保持函数签名不变（隐藏测试会直接调用这些 stub），可以按需要添加私有辅助函数。其余代码均已提供：评分器会用干净副本替换这些给定文件，因此请勿为本作业修改代码库的其他部分。

## 运行

```bash
uv run pytest                                    # 公开测试套件

# 第 2 部分：在 Number Search 上进行同步 REINFORCE 和 GRPO
uv run python scripts/train_sync.py  --config configs/p2a_number_search_reinforce.yaml
uv run python scripts/train_sync.py  --config configs/p2b_number_search_grpo.yaml

# 第 3 部分：异步 GRPO
uv run python scripts/train_async.py --config configs/p3_number_search_async.yaml

# 第 4 部分：消融实验
uv run python scripts/train_sync.py  --config configs/p41_number_search_grpo_g2.yaml
uv run python scripts/train_async.py --config configs/p43_number_search_async_stale4_iw.yaml
uv run python scripts/train_async.py --config configs/p43_number_search_async_stale4_noiw.yaml

# 第 5 部分：在 TextCraft-Synth 上进行异步 GRPO，然后在两个验证集上评估最终 checkpoint
uv run python scripts/train_async.py --config configs/p5_textcraft_grpo.yaml
uv run python scripts/evaluate_checkpoint.py --config configs/p5_textcraft_grpo.yaml \
    --model-path <final_sampler_path> --k 4
uv run python scripts/evaluate_checkpoint.py --config configs/p5_textcraft_grpo.yaml \
    --model-path <final_sampler_path> --k 4 --tasks data/textcraft_val_hard.jsonl

# 将第 5 部分的最终 adapter 导出到公开 Hugging Face 仓库（先执行 hf auth login）
uv run python scripts/export_checkpoint.py --run p5_textcraft_grpo

# 完成全部七次运行后：检查运行结果并写入 submission.zip
uv run python scripts/package_submission.py
```

`<final_sampler_path>` 位于 `runs/p5_textcraft_grpo.checkpoints.json` 中；省略 `--model-path` 即可使用相同设置评估 base model。评分器无法读取你的 `tinker://` checkpoint，因此第 5 部分的 adapter 会以公开 Hugging Face 仓库的形式提交，名称为 `<hf-username>/scottyrl-a3-p5_textcraft_grpo`：`scripts/export_checkpoint.py` 会创建该仓库，并将 id 作为 `hf_model_id` 写入 checkpoint 文件；`scripts/package_submission.py` 会将其写入 `submission/manifest.json`，并在缺少它时拒绝打包。若要自行评估导出的 adapter，请使用 vLLM 提供服务，并向 `scripts/evaluate_checkpoint.py` 传入 `--server http://localhost:8000/v1 --model a3`（确切命令位于该脚本的 docstring 中）。任何配置字段都可以使用 `--set section.key=value` 覆盖（可重复传入），例如使用 `--set train.num_steps=10 --set train.run_name=probe_p2b` 进行低成本探测。必需运行请保留提供的 `run_name` 值：`scripts/package_submission.py` 和评分器会严格查找 `configs/required_runs.yaml` 指定的文件名。

## 恢复运行

一次运行写入的所有内容都以 `train.run_name` 命名。每步日志写入 `runs/<run_name>.jsonl`，捕获步骤的轨迹写入 `runs/<run_name>_trajectories/step_XXXXXX/`，Tinker 上保存的内容列在 `runs/<run_name>.checkpoints.json` 中，最后一个可用 checkpoint 记录在 `runs/<run_name>.resume.json` 中。为保护这些结果，已经有输出的 `run_name` 会被拒绝。错误信息会提供三种处理方式：使用 `--set train.run_name=<name>` 选择新的名称以进行新实验；使用 `--set train.resume=true` 在崩溃后继续同一运行；或者删除 `runs/` 下该运行的文件，从头重做。

恢复依赖运行过程中保存的 checkpoint。每经过 `train.resume_save_every` 个 optimizer step（提供的配置为 10），系统会保存权重和 optimizer 状态；完成保存 checkpoint 的 step 并写入日志后，该 checkpoint 才可用。

恢复时，训练会从最后一个可用 checkpoint 继续。该 checkpoint 之后写入的日志记录和轨迹目录会被删除，相应步骤会重新训练并记录，因此最终日志中的每个 step 仍恰好出现一次。除此之外，恢复运行会像从未中断一样继续：step 编号保持绝对值，不会重复 step-0 基线评估，周期性评估仍按原节奏进行（例如在 `eval_every: 10` 时从 step 45 恢复，下一次会在 step 50 评估）。唯一可见差异是 snapshot 名称。崩溃进程可能已经在服务上使用了普通名称，因此恢复进程会添加 `_r1` 后缀（`step_000041_r1`，第二次恢复后为 `_r2`）。

## 训练后端

`skyrl_tinker_modal/README.zh-CN.md` 介绍了如何为各部分在 Modal 上部署 SkyRL Tinker server。设置 `export TINKER_BASE_URL=<endpoint>`，即可让训练、评估和导出脚本指向你的部署。
对于并行的 Number Search 实验，使用 `SKYRL_APP_NAME` 部署不同的 app，并为每个 app 运行一个 trainer。部署命令见[并行运行不同部分](skyrl_tinker_modal/README.zh-CN.md#并行运行不同部分)。
