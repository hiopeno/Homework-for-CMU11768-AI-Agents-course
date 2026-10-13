# 作业 3：使用强化学习训练 Agent

11-768 AI Agents，CMU，2026 年秋季学期。

## 概述

在本作业中，你将协助构建 `scottyrl`，这是一个用于通过强化学习训练 agent 的最小化训练框架。具体来说，你将：

- 构建轨迹处理与优势计算流水线；
- 实现同步 REINFORCE 和 GRPO 训练；
- 实现能够处理陈旧 rollout 的异步 GRPO；
- 通过受控消融实验比较不同算法和系统设计；
- 在 TextCraft-Synth 上训练并评估一个 agent。

**请尽早开始！** 与之前的作业相比，本作业更重，无论是实现工作量，还是等待各种训练运行完成所需的时间，预计都会更长。

## Tinker API 兼容性

`scottyrl` 构建在 Tinker API 之上，并且与其完全兼容。建议你查看其文档，并在完成本作业的过程中参考文档，以熟悉 Tinker SDK 提供的训练和采样原语：https://tinker-docs.thinkingmachines.ai/tinker/quickstart/。

面向 Tinker API 可以带来很大的灵活性，因为它将高层的 RL 框架与底层的分布式 GPU 执行和状态解耦。前者包括训练循环、agent harness、环境、数据处理、奖励设计和优势计算等；后者则被封装在一个实现受限作用域 API（如 `forward_backward`、`optim_step`、`sample` 等）的服务中。

重要的是，这使得框架具有模块化和高度灵活性：`scottyrl` 可以使用任何实现了类似 Tinker 服务的训练后端。例如：

托管服务（需要 API 额度）：

- [Thinking Machines 官方 Tinker 服务](https://tinker-docs.thinkingmachines.ai/tinker/)
- [Fireworks 的 Tinker 兼容训练服务](https://docs.fireworks.ai/fine-tuning/training-api/introduction)

自托管服务（自备计算资源，并在自己的 GPU 节点上托管 Tinker 兼容后端服务）：

- [SkyRL 的 Tinker 兼容训练服务](https://docs.skyrl.ai/docs/tinker/overview)（支持全参数微调和 LoRA）
- [Miles 的 Tinker 兼容训练服务](https://github.com/radixark/miles/issues/3284)：实验性支持，目前仅支持 LoRA

在作业 3 中，你将使用 SkyRL 的开源 Tinker 服务实现，并使用 [Modal](https://modal.com/) 提供计算资源。

如果你希望在期末项目中使用该框架或基于它进行构建，也可以根据自己拥有的计算资源或 API 额度，自由尝试其他服务选项。

## 使用 SkyRL 和 Modal 自托管

请查看[这里](skyrl_tinker_modal/README.md)的说明，为各个问题设置并配置后端训练服务。

在开始不同问题的训练之前，务必先阅读这份说明，因为我们针对不同的训练运行推荐了不同的配置。

## 环境设置

使用 [uv](https://docs.astral.sh/uv/) 和 Python 3.12 安装项目：

```bash
uv python install 3.12
uv sync
uv run pytest
```

在初始代码中，25 个测试里有 12 个会通过，其余测试会因 `NotImplementedError` 失败。随着你逐步完成实现，通过的测试数量会增加。所有命令都应从仓库根目录通过 `uv run` 执行。测试以及完整的数据流水线都可以在本地运行，不需要 GPU 或后端服务。

准备启动训练运行时，请为对应部分部署 SkyRL 服务，并按照[自托管指南](skyrl_tinker_modal/README.md)中的说明设置 `TINKER_BASE_URL`。共有三个启动脚本，分别对应不同规模的运行；下面每一部分都会说明应使用哪个脚本。每个服务器一次只能运行一个任务，但三个服务器彼此独立，因此可以按照[并行运行各部分](skyrl_tinker_modal/README.md#running-parts-in-parallel)的说明同时运行不同部分。开始训练前，务必先让代码通过测试：服务器从某个运行连接的那一刻起就会产生 GPU 费用，因此一个本可以在本地发现的崩溃也会带来实际成本。

本作业的所有训练运行都使用带 LoRA（rank 32）的 `Qwen/Qwen3.5-4B`。每个必需运行都已经在 [configs/](configs/) 中提供了现成配置。

## 环境

你将在两个环境中训练 agent。在这两个环境里，agent 每一轮通过发送一个工具调用来行动，工具调用写作 <tool>name(args)</tool>，然后接收一条文本观察结果。一个 episode 会在任务解决、agent 调用 `finish()`（TextCraft-Synth）或达到步数上限时结束。奖励是二元的：成功为 1，失败为 0。

**Number Search** 是一个规模小、成本低的环境，你将在作业的大多数部分使用它。agent 必须通过唯一的工具 `guess(n)` 找到 [low, high] 中的隐藏数字，该工具会回复“太低”“太高”或“正确”。agent 可以进行的猜测次数为 ceil(log_2(high-low+2))：这是精确的最坏情况二分查找预算，其中包括最后一次正确猜测，没有额外余量。该环境包含 50,000 个训练任务和 1,000 个验证任务。所有部分都使用同一个 `xml_toolcall` agent。推理由 YAML 中的 chat-template 设置控制：Number Search 使用 `model.enable_thinking: false`，TextCraft 使用 `true`。两者都设置 `model.preserve_thinking_history: true`，以保留之前的 thinking block，包括空 block。

**TextCraft-Synth** 是一个更难的制作环境，你将在最后一部分使用它。agent 从一个初始物品栏和一组目标物品开始，并拥有四个工具：`craft`、`get_info`、`view_inventory` 和 `finish`。必须严格遵循配方，原材料只能来自初始物品栏。只有当 agent 制作出所有目标物品，并在步数耗尽前调用 `finish()` 时，才会获得奖励。该环境包含 852 个训练任务、213 个中等难度验证任务，以及 136 个用于第 5 部分测试泛化能力的高难度验证任务。该环境是论文 [Recursive Agent Optimization (RAO)](https://arxiv.org/abs/2605.06639) 中首次介绍的版本的一个变体。

这两个环境的训练运行有一个重要差异：Number Search 会禁用 chain-of-thought（CoT），而 TextCraft-Synth 会启用 CoT，因此 TextCraft 运行会长得多。

数据集位于 [data/](data/)（参见 [data/README.md](data/README.md)）。请不要编辑或重新生成这些数据集。

## 每次运行的预计成本和时间

这两个环境的预算差异很大，因此请据此安排训练运行：

| 环境 | 每次运行的大致成本 | 每次运行的大致时间 | 预期用途 |
|---|---:|---:|---|
| Number Search | 约 $10 | 约 40 分钟 | 便宜且快速：用于调试、测试实现和消融实验 |
| TextCraft-Synth | 约 $100 | 6–8 小时 | 最终部分只进行一次 **hero run** |

**将 TextCraft-Synth 视为 hero run。** 在使用 Number Search 对 trainer 进行端到端调试和验证之前，不要启动它。不要将 TextCraft-Synth 用于调试或反复尝试。运行开始后，请持续关注指标和轨迹（参见[标准诊断图表](#标准诊断图表)和[健全性警告](#健全性警告)），以便尽早发现错误，并在错误运行继续消耗预算前将其停止。

## 代码库概览

建议从 `scottyrl/types.py` 开始阅读，它定义了在框架中流动的对象：

1. agent（`scottyrl/agents/xml_tool_call.py`）在一个 episode 中与环境（`scottyrl/rollout.py`）交互。这会产生一个 `Trajectory`：记录 episode 中每次 LLM 调用的 token 空间信息，以及最终奖励。
2. 数据流水线（`scottyrl/advantages.py`、`scottyrl/datagen.py`）将轨迹转换成 `TokenSequence`：包含 loss mask、采样 logprob 和 advantages 的训练序列。
3. trainer（`scottyrl/train/`）收集 rollout、构建 batch，并通过封装 Tinker 训练 API 的 `TinkerSession`（`scottyrl/tinker_adapter.py`）将 batch 发送给后端。

[README](README.md) 提供了仓库的完整结构图。你需要按照以下顺序实现六个文件：

| 文件 | 需要实现的内容 | 部分 |
|---|---|---|
| `scottyrl/advantages.py` | `compute_group_advantages` | 1 |
| `scottyrl/datagen.py` | `extension_suffix`、`stitch_trajectory`、`trajectory_to_training_sequences` | 1 |
| `scottyrl/train/common.py` | `collect_group`、`groups_to_batch` | 2 |
| `scottyrl/tinker_adapter.py` | `TinkerSession.submit_update`（文件其余部分已提供） | 2 |
| `scottyrl/train/sync.py` | `SyncTrainer` 的 `collect_batch`、`build_batch`、`update`、`evaluate_now`、`train` | 2 |
| `scottyrl/train/async_grpo.py` | `AsyncGRPOTrainer` 的 `generate_group`、`rollout_worker`、`collect_fresh_batch`、`update`、`evaluate_now`、`train` | 3 |

其余代码均已提供，并且评分前会恢复，因此只有你对这六个文件的修改会计入评分。将每个 stub 的 docstring 视为规范。保持函数签名不变；你可以自由添加私有辅助函数。

`tests/` 中的公开测试会检查你的代码是否正确接入框架。通过公开测试并不代表代码完全正确：我们还会使用更全面的隐藏测试集进行评分。

### Chat Template 与序列扩展

提示词使用一个 chat template（`scottyrl/chat_templates/qwen3_5.jinja`）渲染，该模板基于 Qwen3.5 自带模板改编而来。一个 episode 包含多次 LLM 调用，我们希望将整个 episode 训练为一个序列，而不是每次调用对应一个序列。当每次调用的 prompt 都是“前一次 prompt 加上前一次 completion”的精确扩展时，这是可行的（参见 [Sequence Extension](https://tinker-docs.thinkingmachines.ai/tutorials/advanced/sequence-extension/)）。agent 在 token 空间中构建上下文，以保持这种关系。

两个配置字段控制 prompt 的渲染方式：

- `model.enable_thinking`：是否启用 CoT。Number Search 关闭，TextCraft-Synth 开启；
- `model.preserve_thinking_history`：是否在 prompt 中保留早先轮次的 CoT。当它为 `true`（默认值）时，序列扩展通常成立，一个 episode 会成为单个训练序列（但有时仍会被打断，例如模型以 `<|endoftext|>` 而不是 `<|im_end|>` 停止）。当它为 `false` 时，为节省上下文会移除之前的 `<think>` block。这会破坏序列扩展，因此每一轮会成为独立的训练序列。你的数据流水线需要同时处理这两种情况。

### 运行实验

每次运行都由一个包含 `model:` 和 `train:` 部分的 YAML 配置描述。你可以在命令行中添加 `--set section.field=value` 覆盖任意字段，例如 `--set train.lr=2e-5`（完整命令见[先做一次短探测](#先做一次短探测)）。

一次运行由 `train.run_name` 标识，所有内容都写入 `runs/`：

| 产物 | 路径 |
|---|---|
| 每步指标 | `runs/<run_name>.jsonl` |
| Rollout 轨迹 | `runs/<run_name>_trajectories/step_XXXXXX/` |
| 保存的 checkpoint | `runs/<run_name>.checkpoints.json` |

每个日志的第 0 步包含未训练模型的基线评估。提供的配置每五步保存一次轨迹。请阅读这些轨迹！它们是了解 agent 实际行为的最快方式。

必需运行请保留提供的运行名称，因为评分会查找这些名称。如果某个运行名称已经有输出，系统会拒绝覆盖，以避免意外覆盖结果。新实验请选择新的名称，或者传入 `--set train.resume=true` 从最近的 checkpoint 继续执行崩溃的运行。

### 记录到 Weights & Biases（可选）

每次运行都会将指标写入 JSONL 日志，评分也以该日志为准。我们仍建议额外记录到 [Weights & Biases](https://wandb.ai)：它可以提供每个指标的实时图表，便于在长时间运行期间监控指标，并横向比较多个运行。登录一次后，添加 `--set train.wandb_project=<project>` 启用它：

```bash
uv run wandb login
uv run python scripts/train_sync.py --config configs/p2b_number_search_grpo.yaml \
    --set train.wandb_project=11768-a3
```

每次运行都会以其 `run_name` 出现在 W&B 中，恢复运行会继续使用同一个 W&B run。如果 W&B 无法启动（例如尚未登录），运行会打印 `[metrics] wandb disabled` 并仅继续写入 JSONL 日志，因此请在启动时留意这行输出。

### 标准诊断图表

每个实验报告部分（第 2 部分、第 3 部分、第 4 部分的每个子部分以及第 5 部分）都应为其中讨论的每个训练运行，绘制以下指标随训练步数变化的图表：

- 奖励质量：`train/success_rate`、`eval/success_rate`，以及所有已记录的 `train/pass@*` 和 `eval/pass@*`；
- 优化：`train/grad_norm`；
- 策略漂移：`train/importance_weight_mean`、`train/importance_weight_max` 和 `train/importance_weight_std`；
- 数据健康度：`train/frac_degenerate_groups` 和 `train/entropy_proxy`。

你可以将相关指标和运行合并到易读的多面板图中。为每条 pass@ 曲线标注其完整 key：训练和评估通常使用不同的 k，因此它们是不同的量。

### 时间指标

每条训练记录还包含自上一条训练记录以来这段时间间隔的时间指标，单位为秒。这些指标由已提供的代码记录，因此你无需实现它们，也不需要将它们纳入标准诊断图表。请使用它们理解一次运行的时间花在哪里，并尽早发现性能下降。

| 指标 | 衡量内容 |
|---|---|
| `train/step_time_s` | 一个训练 step 的墙钟时间长度 |
| `train/update_time_s` | 优化器更新所花时间，包括定期的可恢复 checkpoint 保存 |
| `train/learner_wait_s` | step 时间减去更新时间和 sampler snapshot 保存时间，即 learner 在其他环节等待的时间 |
| `train/learner_wait_frac` | `learner_wait_s` 占 `step_time_s` 的比例 |
| `train/rollout_time_s` | 该 step 内完成的训练 episode 中，从最早 episode 开始到最晚 episode 结束的时间 |
| `train/episode_time_mean_s` | 这些 episode 的平均耗时 |

## 背景知识

### 从 Episode 到梯度

设一个拼接后的 episode 包含 token $t_0,\ldots,t_{n-1}$，并令 $m_j \in \{0,1\}$ 表示目标 token $t_{j+1}$ 是否由策略采样得到，而不是来自 prompt 或环境。设 episode 级别的优势为 $A$，则策略梯度为

$$\nabla_\theta J(\theta) = \mathbb{E}\Big[\, A \sum_{j:m_j=1} \nabla_\theta \log \pi_\theta(t_{j+1}\mid t_{\le j}) \Big].$$

只有策略采样的 token 才应属于 loss。如果 mask 包含观察 token，模型学到的将是预测环境，而不是学习如何行动。

### REINFORCE 与 GRPO

REINFORCE 使用 $A = r$：成功 episode 中的每个采样 token 都会被强化，而失败 episode 不产生贡献。该估计是无偏的，但噪声较大，尤其是在奖励稀疏时。

GRPO 针对同一个任务采样一组包含 $G$ 个 rollout 的组，并使用该组的平均奖励作为 baseline：

$$A_i = r_i - \frac{1}{G}\sum_{k=1}^{G} r_k.$$

减去 baseline 可以消除任务难度的影响，因此只有当某个 rollout 的表现优于策略在该任务上的典型尝试时，它才会被强化。如果一个组中的所有奖励都相同，那么所有 advantage 都为零，该组不包含任何学习信号。我们称这样的组为 *degenerate*，并跳过它。按照 [Dr.GRPO](https://arxiv.org/abs/2503.20783) 的做法，我们不除以组奖励的标准差，也不对参考策略加入 KL 惩罚。

在配置中，`train.algo` 单独决定 advantage 规则，而 `train.loss_fn` 单独决定下面介绍的 token 级 loss。

### 异步训练与重要性加权

同步训练交替进行 rollout 收集和模型更新，因此每个 batch 都是 on-policy 的，但 learner 在 rollout 期间空闲，rollout worker 又在更新期间空闲。异步训练会将两者重叠执行，因此速度更快；但此时一个组可能是由若干次更新之前的旧策略采样得到的。我们通过两种方式处理这种*陈旧性*：丢弃版本差距超过 `train.max_staleness` 的组，并对其余组使用重要性加权进行校正。

对于从 $\pi_{\text{old}}$ 采样、并在 $\pi_\theta$ 下训练的 token，重要性比率为

$$\rho_j = \exp\big(\log \pi_\theta(t_{j+1}\mid t_{\le j}) - \log \pi_{\text{old}}(t_{j+1}\mid t_{\le j})\big).$$

框架支持三种 loss：

- `ppo`，裁剪后的 surrogate：$-\sum_{j:m_j=1} \min(\rho_j A_j,\ \mathop{\text{clip}}(\rho_j, 1-\epsilon, 1+\epsilon)\,A_j)$；
- `importance_sampling`，与上面相同但不进行裁剪：$-\sum_{j:m_j=1} \rho_j A_j$；
- `cross_entropy`，不使用比率的、按 advantage 加权的对数似然：$-\sum_{j:m_j=1} A_j \log \pi_\theta(t_{j+1}\mid t_{\le j})$。

GRPO 运行使用 `ppo`，REINFORCE 运行使用 `importance_sampling`，而 `cross_entropy` 只用于第 4 部分的“无校正”消融实验。

每次更新都会记录 $\rho$ 的统计量。在同步运行中，它们应始终非常接近 1。如果不是这样，很可能说明你的采样 logprobs 被关联到了错误的 token。

## 第 0 部分：热身（0 分）

在编写代码之前，熟悉 agent、rollout 和环境的实现：`scottyrl/agents/xml_tool_call.py`、`scottyrl/rollout.py` 和 `scottyrl/envs/`。

## 第 1 部分：数据流水线（20 分）

在本部分中，你将把 agent 轨迹转换为训练数据。

按照 docstring 实现 `scottyrl/advantages.py` 中的 `compute_group_advantages`，以及 `scottyrl/datagen.py` 中的三个函数。

务必注意 next-token 对齐：包含 $n$ 个 token 的序列有 $n-1$ 个预测目标，因此 `mask`、`logprobs` 和 `advantages` 的长度都是 $n-1$，位置 $j$ 描述目标 `tokens[j+1]`。如果长度为 $c$ 的 completion 接在长度为 $L$ 的 context 后面，那么它对应的位置是 `L-1 : L-1+c`，而不是 `L : L+c`。

进行拼接时，只要下一次调用的 prompt 是当前已累积内容的扩展，就继续扩展当前序列；当它不再是扩展时，就开始一个新的序列。只有 completion token 的 mask 应为 1，并同时写入它们的采样 logprob 和 episode advantage。

使用以下命令测试：

```bash
uv run pytest tests/test_interfaces.py -k "advantages or datagen"
```

该文件中的其他测试属于第 2 部分。`tests/test_envs.py` 中的 `ThinkStrippingPolicy` 会生成一个破坏序列扩展的 episode，适合在本地测试这种情况。

## 第 2 部分：同步 REINFORCE 和 GRPO（25 分）

接下来，你将构建同步训练循环，并使用它训练你的第一个 agent。

**实现以下内容：**

- `scottyrl/train/common.py` 中的 `collect_group` 和 `groups_to_batch`。前者并发运行一组 rollout，后者计算 advantage 并将各组展平为一个训练 batch；
- `scottyrl/tinker_adapter.py` 中的 `TinkerSession.submit_update`。参考 Tinker 文档，理解如何使用其 API；
- `scottyrl/train/sync.py` 中的 `SyncTrainer`：四个小型辅助方法，以及将它们串联起来的 `train` 循环。务必在每次更新后刷新采样策略，使下一批数据保持 on-policy。

使用以下命令测试：

```bash
uv run pytest tests/test_interfaces.py tests/test_contracts.py -k "not async"
```

完成第 3 部分后，剩余的 `test_async_*` 测试也会通过。

使用提供的配置，针对 `serve_number_search_sync.py` 部署运行两个 Number Search 实验。正式运行前，建议先进行一次[短探测](#先做一次短探测)作为健全性检查。

```bash
uv run python scripts/train_sync.py --config configs/p2a_number_search_reinforce.yaml
uv run python scripts/train_sync.py --config configs/p2b_number_search_grpo.yaml
```

`p2a` 是每个任务仅执行一次 rollout 的 REINFORCE，`p2b` 是每个任务执行八次 rollout 的 GRPO。两者都训练 50 步，每步使用 8 个任务；每 10 步在 50 个验证任务上各执行 4 次 rollout 进行评估。

**报告要求：**

- 为两次运行都包含[标准诊断图表](#标准诊断图表)；
- 用 3–5 句话比较 REINFORCE 和 GRPO 的学习速度，并解释这种差异是否符合预期、原因是什么。正常运行的奖励应在 50 步内明显高于 step-0 基线。如果曲线保持平坦，请先调试实现，再撰写分析。

## 第 3 部分：异步 GRPO（25 分）

在本部分中，你将通过生产者—消费者设计减少训练中的空闲时间。Rollout worker 持续使用最新发布的策略生成组，并将这些组推入有界队列。Learner 从队列中取出组，对其训练，然后发布新的策略版本。

按照 docstring 实现 `scottyrl/train/async_grpo.py` 中 `AsyncGRPOTrainer` 的六个学生方法。需要特别注意：

- 每个组都应使用预先捕获的单一 `(policy, version)` 对生成，这样即使在生成过程中途发布了新策略，它携带的版本也仍然正确。陈旧性应由 learner 在消费该组时检查，而不是由 worker 检查；
- 队列有意设置为有界队列。让 [asyncio.Queue](https://docs.python.org/3/library/asyncio-queue.html) 通过 `await queue.put(...)` 和 `await queue.get()` 提供反压，不要轮询 `full()` 或 `empty()`；
- 评估和关闭都不应阻塞训练。每次评估都应针对捕获的 `(policy, version)` 作为后台任务运行，并与训练重叠。当训练结束时，[取消](https://docs.python.org/3/library/asyncio-task.html#task-cancellation)并等待 worker；如果发生任何失败，则取消其他所有任务，让原始错误继续传播，而不是挂起。

使用 `uv run pytest` 测试：此时整个公开测试套件都应通过。`tests/test_contracts.py` 中的 `test_async_*` 测试会用伪造的 rollout 和更新驱动 trainer，因此无需消耗计算资源就可以正确处理并发。

使用提供的配置，针对 `serve_number_search_async.py` 部署运行：异步训练需要为训练和采样分别使用 GPU，因此第 2 部分的服务器不能用于此处。

```bash
uv run python scripts/train_async.py --config configs/p3_number_search_async.yaml
```

`p3` 与 `p2b` 匹配（每步八组、每组八次 rollout，共 50 步），但它使用八个 rollout worker 生成数据，并允许最多 2 个版本的陈旧性。运行结束时，`train()` 返回的汇总会打印出来，并且也会写入 `runs/p3_number_search_async.jsonl` 的最后一条记录，字段分别为 `summary/train_steps`、`summary/groups_consumed` 和 `summary/groups_dropped_stale`。你将在第 4.2 部分中用到 `summary/groups_dropped_stale`。

**报告要求：**

- 为 `p3` 包含[标准诊断图表](#标准诊断图表)。在分析中比较重要性权重的均值、最小值、最大值和标准差；均值接近 1 可能掩盖重尾分布；
- 使用 `train/step_time_s`（或相邻训练记录 `time` 字段之差）比较 `p2b` 的每步墙钟时间，并量化加速比；
- 从轨迹中选择一个早期、一个中期和一个后期训练 rollout（给出 step、task id，以及文件头中的 `policy_version`）。描述 agent 行为至少两种具体变化，例如它如何使用反馈、是否采用二分或扫描，以及如何从无效调用中恢复。

## 第 4 部分：Number Search 消融实验（20 分）

现在使用你的 trainer 进行三项小型研究。为每项研究在报告中写一个简短章节，并包含所要求的图表和一段分析。使用提供的配置，不要修改。

### 4.1 组大小

比较 `p2a`（$G=1$，REINFORCE）、`p41`（$G=2$，GRPO）和 `p2b`（$G=8$，GRPO）。唯一新增的运行是 `p41`，它使用与第 2 部分相同的部署：

```bash
uv run python scripts/train_sync.py --config configs/p41_number_search_grpo_g2.yaml
```

**报告要求：** 为三次运行都包含[标准诊断图表](#标准诊断图表)。使用 `eval/success_rate` 和 `train/frac_degenerate_groups` 解释较小的 group 如何影响 advantage 的方差，以及每个 step 中有用序列的数量。对于 REINFORCE，将 degenerate fraction 报告为 N/A：它没有 group baseline，因此其记录值 0 没有实际意义。请记住，`p2a` 在 advantage 规则和 loss 上也与 GRPO 运行不同，因此这不是纯粹的组大小消融。

### 4.2 同步训练与异步训练

**报告要求：** 包含 `p2b` 和 `p3` 的[标准诊断图表](#标准诊断图表)。复用第 3 部分的墙钟时间分析，并增加每次运行最终的 `eval/success_rate`（一个小表格即可）。解释加速来自哪里，以及在陈旧数据上训练是否付出了样本质量代价。使用重要性权重统计量和 `p3` 日志最后一条记录中的 `summary/groups_dropped_stale` 计数作为证据。

### 4.3 更高陈旧性下的重要性加权

运行两个异步实验。两者唯一的差别是 loss，并且都使用与第 3 部分相同的部署：

```bash
uv run python scripts/train_async.py --config configs/p43_number_search_async_stale4_iw.yaml
uv run python scripts/train_async.py --config configs/p43_number_search_async_stale4_noiw.yaml
```

两者都允许最多 4 个版本的陈旧性，以便明显观察其影响。`iw` 运行使用 PPO，而 `noiw` 使用完全忽略重要性比率的 advantage-weighted cross-entropy。

**报告要求：** 为两次运行都包含[标准诊断图表](#标准诊断图表)。

## 第 5 部分：TextCraft-Synth（10 分）

在最后一部分中，你将使用异步 GRPO trainer 在 TextCraft-Synth 上进行一次训练，然后评估 agent 在中等难度和高难度验证集上的表现。

> **这是一次 hero run（约 $100，耗时 6–8 小时）。** 只有在异步 trainer 已经能够在 Number Search 上正常工作后才启动它，并监控指标，以便出现问题时尽早停止。参见[每次运行的预计成本和时间](#每次运行的预计成本和时间)。

使用提供的配置，不要修改。该配置启用了 thinking，每轮最多 512 个 token。50 个 step 中的每个 step 都会训练 16 组、每组 8 次 rollout；这些 rollout 由 32 个 worker 生成，最多允许 4 个版本的陈旧性，并且每 10 步评估一次。TextCraft episode 远长于 Number Search episode，大部分时间都花在采样上，因此本部分使用专门的、更大规模部署（`serve_textcraft.py`）：

```bash
uv run python scripts/train_async.py --config configs/p5_textcraft_grpo.yaml
```

使用同一个部署，在中等难度和高难度验证集上评估最终 checkpoint 和 base model；每个任务执行 4 次 rollout。checkpoint 路径位于 `runs/p5_textcraft_grpo.checkpoints.json` 的 `final_sampler_path` 中：

```bash
# 微调后的模型
uv run python scripts/evaluate_checkpoint.py --config configs/p5_textcraft_grpo.yaml \
    --model-path <final_sampler_path> --k 4
uv run python scripts/evaluate_checkpoint.py --config configs/p5_textcraft_grpo.yaml \
    --model-path <final_sampler_path> --k 4 --tasks data/textcraft_val_hard.jsonl

# base model（不传 --model-path）
uv run python scripts/evaluate_checkpoint.py --config configs/p5_textcraft_grpo.yaml --k 4
uv run python scripts/evaluate_checkpoint.py --config configs/p5_textcraft_grpo.yaml --k 4 \
    --tasks data/textcraft_val_hard.jsonl
```

将最终 adapter 导出到 Hugging Face，供课程方评估。该脚本会通过 `serve_textcraft.py` 部署下载 checkpoint，因此部署必须保持运行。如果已经停止，请使用相同命令重新部署，并让 `TINKER_BASE_URL` 继续指向它；checkpoint 会保存在其 volume 中。训练后立即导出可以省去额外启动服务的时间。第一次使用时需要登录：

```bash
uv run hf auth login
uv run python scripts/export_checkpoint.py --run p5_textcraft_grpo
```

脚本会将 LoRA adapter 上传到名为 `<hf-username>/scottyrl-a3-p5_textcraft_grpo` 的公开模型仓库，并在 checkpoint 文件中将其记录为 `hf_model_id`。如需更新已有导出，可添加 `--exist-ok`；在成绩发布前，请保持该模型仓库公开。

**报告要求：**

- 包含 `p5` 训练运行的[标准诊断图表](#标准诊断图表)；
- 使用四行表格展示微调模型和 base model 在两个数据划分上的结果，每行包含 `success_rate` 和 `pass@4`；
- 用几句话分析本次运行，并描述结果。

## 提示与调试

### 先做一次短探测

在开始长时间运行之前，先使用一个临时名称进行短运行：

```bash
uv run python scripts/train_sync.py --config configs/p2b_number_search_grpo.yaml \
    --set train.num_steps=10 --set train.eval_every=2 --set train.eval_tasks=20 \
    --set train.run_name=probe_p2b
```

注意：如果你只想检查训练是否能够运行，而不执行评估，可以添加 `--set train.eval_every=0`（或者等价地添加 `--set train.eval_tasks=0`）完全禁用评估，包括 step-0 基线。此时日志只包含训练记录，因此请跳过下面检查 1 中关于评估的部分。

然后检查：

1. 日志包含 step 0 的基线评估、每个 step 一条训练记录，以及每两步一次的评估；
2. `step_000005/` 下的轨迹看起来像真实 episode：包含 completion、解析后的工具调用和环境回复。`null` 工具调用表示 completion 解析失败。唯一的例外是最后一轮可能只有 `stop_reason` 而没有 completion：这表示某次采样调用丢失，例如因为超出服务器上下文窗口，此 episode 应计为失败；
3. `train/importance_weight_mean` 保持接近 1，这是同步运行应有的情况；
4. 你可以解释终端输出中的每一行 `[sanity]` 信息。

请注意，短探测只能检查整个流程是否正确连通。10 步不足以判断模型是否真正学会了任务。

### 健全性警告

训练期间，当某些情况看起来不正常时，框架会打印 `[sanity]` 警告：熵塌缩、重要性权重偏离 1、梯度尖峰，或者几乎所有 group 都是 degenerate。具体阈值定义在 `scottyrl/metrics.py` 中。

## 报告

提交位于仓库根目录的 `report.pdf`。每一部分要求包含的内容已经在相应章节列出。

请确保图表可读（包括坐标轴、图例和标题），并讨论你展示的每一条曲线。

## AI 使用说明

提交根目录下的 `AI_USAGE.md`，详细说明你在本作业中使用任何 AI 技术的情况。列出你使用的所有工具，并清楚描述每个工具的使用方式。如果没有使用 AI 辅助，也请在该文件中声明。

将文件放在仓库根目录，与 `report.pdf` 并列。打包脚本会将它加入 `submission.zip`，并且如果缺少该文件则拒绝打包。

## 提交

完成全部七次必需运行以及第 5 部分的导出后，打包提交：

```bash
uv run python scripts/package_submission.py
```

脚本会打印检查清单并写入 `submission.zip`。压缩包包含六个源文件、配置文件、日志、轨迹、每个必需运行的 checkpoint 文件、`report.pdf`、`AI_USAGE.md`，以及记录第 5 部分 adapter 的 `hf_model_id` 的 manifest。对于不完整的提交，脚本会拒绝打包并告诉你缺少什么。

将 `submission.zip` 上传到 Canvas。迟交天数按课程 syllabus 执行。

## 评分

| 部分 | 分值 |
|---|---:|
| 1 | 20 |
| 2 | 25 |
| 3 | 25 |
| 4 | 20 |
| 5 | 10 |
| | **100** |
