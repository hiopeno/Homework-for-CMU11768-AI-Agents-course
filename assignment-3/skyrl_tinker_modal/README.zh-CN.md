# 使用 SkyRL 和 Modal 自托管训练后端

`scottyrl` 可以连接任何兼容 Tinker 的服务进行训练。本作业使用的是 [SkyRL 的 Tinker server](https://docs.skyrl.ai/docs/tinker/overview)，由你自行部署在 [Modal](https://modal.com/) 的 GPU 上。本目录为作业中的每类运行提供一个启动脚本。每个脚本都会申请该类运行所需的 GPU，并根据训练和采样的需求分配 GPU。

| 启动脚本 | 用于 | GPU | 布局 |
|---|---|---|---|
| `serve_number_search_sync.py` | 第 2 部分（`p2a`、`p2b`）和第 4.1 部分（`p41`） | 2 张 H100 | 共享：两张 GPU 都用于训练和采样 |
| `serve_number_search_async.py` | 第 3 部分（`p3`）和第 4.3 部分（`p43_*`） | 2 张 H100 | 分离：1 张用于训练，1 张用于采样 |
| `serve_textcraft.py` | 第 5 部分（`p5`）、该部分的评估和导出 | 4 张 H100 | 分离：1 张用于训练，3 张用于采样 |

不同布局的原因是两种 trainer 对 GPU 的使用方式不同。同步训练交替进行采样和更新，因此让两者共享 GPU 可以使两张 GPU 在两个阶段都保持繁忙（SkyRL 会在更新期间让采样引擎休眠，更新后再唤醒）。异步训练会在更新的同时进行采样，共享 GPU 的服务器无法做到这一点，因此训练和采样各需要一张 GPU。第 5 部分同样使用异步训练，而 TextCraft-Synth episode 较长，大部分时间都花在采样上，所以它的四张 GPU 中有三张分配给采样器。

三个部署共享的部分（容器镜像、SkyRL 版本和优化器设置）位于 `skyrl_server.py`。三个服务都使用 SkyRL 的 Megatron 后端进行训练，这是 SkyRL 支持 Qwen3.5 的后端，并使用 vLLM 进行采样。镜像固定了 [我们的 SkyRL fork](https://github.com/adityasoni9998/SkyRL) 的一个 revision，其中包含 Megatron token-batched 输出顺序修复和仅保存 adapter 的 sampler checkpoint 支持，因此保存的 LoRA snapshot 是 adapter，而不是整个模型的副本。这些改动已经包含在固定的源码中，构建镜像时不会应用本地 patch。你通常不需要编辑这些文件。

## 一次性设置

创建 Modal 账户，然后安装并认证 Modal CLI：

```bash
uv sync --extra modal
uv run modal setup
```

Modal 是可选依赖，因此之后执行普通的 `uv sync` 会将其移除。如果 `uv run modal` 不再工作，请再次执行一次 `uv sync --extra modal`。

下面的所有命令都应从本目录执行：

```bash
cd skyrl_tinker_modal
```

第一次部署时会构建包含 SkyRL、Megatron 及其 CUDA 依赖的容器镜像。这需要下载几十 GB 的内容，预计耗时 20 分钟或更久。构建过程会使用一张 H100，因此成本大约等于该 GPU 使用同样时长的费用。该镜像由三个脚本共享，并由 Modal 缓存，所以只需等待和付费一次。如果想提前构建并验证镜像是否正常，可以运行可选的 smoke test；它会在需要时构建镜像，然后再使用一张 H100 运行几分钟：

```bash
uv run modal run serve_number_search_sync.py::smoke_install
```

## 运行某个部分

**1. 为当前部分部署服务器。** 例如，第 2 部分：

```bash
uv run modal deploy serve_number_search_sync.py
```

Modal 会打印部署 URL。除一次性的镜像构建外，部署本身不产生费用：服务器会缩容到零，只有训练运行连接后才会分配 GPU。

**2. 让 `scottyrl` 指向服务器并开始训练。** 从仓库根目录执行：

```bash
export TINKER_BASE_URL=https://<your-workspace>--scottyrl-ns-sync-tinker-server.modal.run
uv run python scripts/train_sync.py --config configs/p2b_number_search_grpo.yaml
```

复制 `modal deploy` 打印的准确 URL。它通常具有上述形式。

第一个请求会启动容器、加载模型并启动采样引擎。在 step-0 评估开始前，预计需要等待几分钟。之后对热启动服务器的运行会立即开始。

**3. 完成后停止服务器。** 空闲服务器会在五分钟后自动关闭，但这段时间仍会产生费用。若要立即停止：

```bash
uv run modal app stop scottyrl-ns-sync
```

停止操作会移除部署，因此下次运行前需要再次执行 `uv run modal deploy ...`。由于镜像已经缓存，这只需几秒钟，并且 URL 保持不变。你的 checkpoint 存储在 volume 中，不会受到影响。

应用名称为 `scottyrl-ns-sync`、`scottyrl-ns-async` 和 `scottyrl-textcraft`。你可以在 Modal dashboard 中查看正在运行的应用及其当前费用。

## 并行运行不同部分

三个启动脚本是彼此独立的 Modal app，各自拥有 URL、GPU 和 checkpoint volume，因此无需额外设置即可同时运行不同部分。只有当你想在同一个启动脚本上运行两个任务时，才需要设置第二个应用名称，例如同时运行都使用同步服务器的 `p2a` 和 `p41`。部署任意 Number Search 启动脚本时设置 `SKYRL_APP_NAME`：

```bash
SKYRL_APP_NAME=scottyrl-p2a uv run modal deploy serve_number_search_sync.py
SKYRL_APP_NAME=scottyrl-p41 uv run modal deploy serve_number_search_sync.py
```

每个 app 都有自己的 endpoint 和 checkpoint/database volume。在不同终端中，将 `TINKER_BASE_URL` 设置为对应部署的 URL，并在每个 app 上启动一个训练运行。不设置 `SKYRL_APP_NAME` 时，使用默认 app 名称。恢复运行时使用相同的 app 名称；完成后使用 `uv run modal app stop <app name>` 停止每个 app。

## Trainer 和推理设置

每个 `serve_*.py` 文件都会将 [skyrl_server.py](skyrl_server.py) 中的 `BASE_BACKEND_CONFIG` 与该 profile 的覆盖配置合并。下面的设置仅供参考；对于所有必需运行，你通常不需要修改它们。服务器设置位于这些文件中，而不是客户端训练 YAML 中；修改后只有重新部署才会生效（修改 `BASE_BACKEND_CONFIG` 会影响三个服务器）。

以下是 SkyRL 后端的配置项：

| 配置项 | 含义及当前值 |
|---|---|
| `trainer.placement.policy_num_nodes` | 使用的 GPU 节点数：`1` |
| `trainer.policy.megatron_config.{tensor,pipeline,context}_model_parallel_size` | Megatron 的模型、流水线和上下文并行度：`1`（4B 模型使用 LoRA 后可以放入一张 GPU，因此这里只使用数据并行） |
| `trainer.placement.policy_num_gpus_per_node` | 分配给 Megatron 训练后端的 GPU 数：同步 RL + Number Search 为 `2`，其他情况为 `1`。 |
| `trainer.placement.colocate_all` | 训练器和推理引擎是否共享 GPU：同步 RL 中训练和采样交替进行，因此为 `True`；异步 RL 为 `False`。 |
| `generator.inference_engine.num_engines` | 部署的 vLLM 推理引擎数：同步 RL + Number Search 为 `2`，异步 RL + Number Search 为 `1`，TextCraft 为 `3`。我们使用数据并行，因此该数等于推理 GPU 数。 |
| `trainer.policy.language_model_only` | 仅训练文本 backbone。由于 Qwen3.5 是多模态 LLM，而本作业只处理文本任务，因此为 `True`。 |
| `generator.inference_engine.language_model_only` | vLLM 仅加载文本 backbone；由于任务是纯文本，因此为 `True`。 |
| `trainer.policy.megatron_config.lora_config.merge_lora` | 是否将 LoRA adapter 合并到 base model：`False`（为采样和 adapter checkpoint 保持分离）。 |
| `trainer.policy.optimizer_config.max_grad_norm` | 将梯度范数裁剪到该值：`1.0`。 |
| `trainer.policy.optimizer_config.adam_betas` | Adam 的一阶和二阶矩衰减率：`[0.9, 0.95]`。 |
| `trainer.policy.optimizer_config.weight_decay` | Adam 中的权重衰减：`0.0`（禁用）。 |
| `trainer.use_expandable_segments` | 训练 worker 使用 PyTorch 的 expandable-segments 分配器，以减少内存碎片：`True`。 |
| `trainer.max_tokens_per_microbatch` | 每个训练 micro-batch 的最大 token 数，用于[动态批处理](https://docs.nvidia.com/nemo/rl/latest/design-docs/sequence-packing-and-dynamic-batching.html)：`32768`（SkyRL 按 token 数将可变数量的完整序列装箱到每个 micro-batch，而不是使用固定序列数；这是软上限，超过该值的单个序列会单独运行在一个 micro-batch 中）。 |
| `trainer.fused_lm_head_logprob` | 在输出层内分块计算 logprob，而不保存完整词表 logits：所有部分均为 `True`（在该 micro-batch 大小下，未融合的反向传播会耗尽 GPU 显存）。 |
| `trainer.logprobs_chunk_size` | 每个 logprob 分块的 token 数：`1024`；Number Search 服务器显式设置，TextCraft 使用 SkyRL 默认值。 |
| `generator.inference_engine.gpu_memory_utilization` | vLLM 可使用每张推理 GPU 显存的比例，包括权重、KV cache 和激活值：共享布局为 `0.7`（为训练器留出空间），其他布局为 `0.85`。 |
| `generator.inference_engine.engine_init_kwargs.max_model_len` | vLLM 上下文窗口（prompt token + 输出 token 的最大值）：Number Search 为 `32768`，TextCraft 为 `65000`。超过该限制的请求会被拒绝；框架随后会以奖励 0 放弃该 episode，运行继续执行（`scottyrl/rollout.py`）。 |
| `generator.inference_engine.max_num_batched_tokens` | 每个 vLLM 引擎单次推理 batch 允许的最大 token 数：所有服务器配置均为 `50000`（50K，与训练 micro-batch 上限相互独立）。 |
| `generator.inference_engine.max_num_seqs` | 单次推理 batch 允许的最大序列数：`256`。 |
| `generator.inference_engine.enable_prefix_caching` | 启用 vLLM 的 prefix caching 以加快推理：`True`。 |
| `generator.inference_engine.use_expandable_segments` | vLLM 进程使用相同的分配器设置：`False`。 |
| `generator.inference_engine.enforce_eager` | 禁用 [CUDA graphs](https://github.com/vllm-project/vllm/issues/4449)，使用 eager PyTorch 执行推理（可以减少内存使用，但速度更慢）：`False`。 |
| `generator.inference_engine.engine_init_kwargs.gdn_prefill_backend` | Qwen3.5 的 Gated-DeltaNet prefill 使用的 kernel backend：`"triton"`。 |
| `generator.inference_engine.engine_init_kwargs.compilation_config` | CUDA graph 配置：`{"cudagraph_mode": "FULL_DECODE_ONLY"}`（仅解码）。 |
| `generator.inference_engine.enable_ray_prometheus_stats` | 通过 Ray 的 Prometheus 集成暴露推理统计信息：`True`。 |
| `generator.inference_engine.engine_init_kwargs.enable_mfu_metrics` | 记录推理期间的模型 FLOPs 利用率：`True`。 |

Tinker 请求会提供学习率；上面的优化器参数在服务器端配置。

三个部分的设置如下：

| Profile | GPU 分配 | vLLM 显存比例 | 上下文限制 | 融合 logprob | 原因 |
|---|---|---:|---:|---|---|
| Number Search sync | 共享 2 张 H100 | `0.7` | `32768` | `True` | 训练和采样交替进行；为两者留出显存。 |
| Number Search async | 1 张训练 H100 + 1 张推理 H100 | `0.85` | `32768` | `True` | 使用独立 GPU 并发训练和采样。 |
| TextCraft | 1 张训练 H100 + 3 张推理 H100 | `0.85` | `65000` | `True` | 更长的 episode 需要更大的上下文和采样能力。 |

如果修改 GPU 分配，请确保它与 Modal 的 GPU 申请保持一致。三个服务器还会申请 8 个 CPU 和 32 GiB 内存，服务器超时为 12 小时，空闲缩容窗口为 5 分钟。

## 费用

按照 Modal 约每张 H100 每小时 4 美元的标价，容器运行期间，每个 Number Search 服务器的费用约为每小时 8 美元，TextCraft 服务器约为每小时 16 美元。请查看 [Modal 定价](https://modal.com/pricing) 获取当前价格。以下习惯有助于减少费用：

- 部署前先在本地让代码通过测试。测试不需要 GPU。
- 每次完整运行前先进行一次[短探测](../ASSIGNMENT.zh-CN.md#先做一次短探测)。探测在第 2 步崩溃，比正式运行在第 40 步崩溃便宜得多。
- 暂时离开时停止 app，绝不要让 TextCraft 服务器整夜运行。
