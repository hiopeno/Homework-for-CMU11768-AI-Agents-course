# 作业 1：构建 Agent Harness

Agent harness 为语言模型（生成概率字符串）提供了观察环境并在环境中行动的接口。构建 agent harness 最流行的框架之一是 [ReAct](https://arxiv.org/abs/2210.03629)：它在语言模型的提示中交错放入对应于环境观察、推理/思维链和 agent 行动的文本。在本作业中，你将使用 ReAct 框架构建一个 harness。

在第 1 部分中，你将实现基础 [`Agent`](./src/assignment/agent/base.py) 的 ReAct 循环，并将其实例化为在终端中解决软件问题的 [`CodeAgent`](./src/assignment/agent/code_agent.py)。你的 ReAct 循环应当是抽象的，并能够应用于多个领域，以便你理解 agent harness 的通用结构。你将实现让编码 agent 观察并操作终端环境的机制，并应用可复用的 agent 技能来解决任务。随后，你将让这个编码 agent 修复一个棋类游戏应用中的问题。

在第 2 部分中，你将处理编码 agent 在解决更复杂软件问题时如何使用上下文。你将实现一个简单的上下文压缩方法，用来管理 agent 使用的语言模型上下文窗口大小。

在第 3 部分中，你将把构建好的基础 `Agent` 实例化为 `ChessAgent`，让它在你用 `CodeAgent` 修复的应用上进行下棋。你将为这个 agent 实现工具规范，使语言模型能够与应用中运行的、由基于规则的 bot 操控的棋局对弈（你也可以旁观）。你还将探索程序化工具调用：它把程序的能力带给 agent 行动，使 agent 可以通过执行 Python 代码来实现更复杂的策略。

三部分的关系如下：

```text
                SWE-Bench issue
                      |
                      v
buggy chess app -> CodeAgent -> fix.patch -> repaired chess server
                                              ^
                                              |
                                    ChessAgent tools
```

## 环境设置

安装 [uv](https://docs.astral.sh/uv/)，然后运行：
```bash
make setup
```
该命令会设置 Python 环境（使用 `uv`），下载并安装相关依赖。如果缺少固定版本的 `chess_app` 源码（agent 要处理的任务所需），或其提交版本不正确，该命令会失败。

本作业会使用 [Modal](https://modal.com/)，这是一个运行代码的云平台。agent 所处的环境是远程 Modal sandbox，agent 执行动作时运行的代码也在其中。首先设置 Modal 账户（如果还没有），并按照课程提供的说明获取 Modal 计算额度。然后运行：
```bash
uv run modal setup
```
登录并在环境中完成 Modal 设置。

完成 Modal 设置后，运行：
```bash
cp .env.example .env
```
创建包含环境机密信息的文件，主要是 LLM API 配置。若你已注册本课程，课程会提供获取 LLM 服务商额度的说明。请按照说明生成 API 密钥，并确保正确配置服务的 base URL。

```dotenv
OPENAI_BASE_URL=<OpenAI-compatible base URL>
OPENAI_API_KEY=...
OPENAI_MODEL=deepseek-flash
OPENAI_MAX_RETRIES=5
```
默认使用上面所示的 deepseek-flash 模型。在明确要求时应使用其他模型。如果额度允许，也可以探索同一服务商的其他模型，但本作业预期使用此模型完成。

我们将任何会使用 API 额度（Modal 或 LLM 服务商）的活动称为 _计费活动（billable）_。作业过程中会多次运行计费评估。建议监控相关控制台的用量，确保合理使用 API 额度。

在运行计费任务前，先验证子模块、Modal 身份验证和模型端点；此命令不会启动 sandbox 或生成 token：
```bash
make doctor
```

（可选）如果想测试所有 Modal 组件是否按预期运行，可以执行：
```bash
make test-modal
make test-chess-modal
```

一般来说，可以使用以下命令检查是否有正在运行且会产生费用的 Modal sandbox：
```bash
modal container list
```
如果环境意外关闭，sandbox 可能仍在运行并继续消耗额度。可以使用 `modal container stop <container ID>` 停止未正确终止的 sandbox。

## 公共测试

`make test` 快速、离线且不计费。起始代码会有意让对应学生 TODO 的测试失败。请用这些里程碑作为指引；如果加入澄清测试，确切的 pytest 数量可能会变化。

通过公共测试并不能证明实现完全正确。私有测试还会覆盖失败时的清理、重复或格式错误的技能、并行棋局调用、传输错误、产物一致性、补丁重放和真实 Modal 集成。

## 规则

- 不要修改 `tests/`、`tasks/`、`chess_app/`。
- 不要修改题目提供的日志/清理逻辑，也不要在子类中重复实现共享的 ReAct 循环。
- 不要硬编码 agent 任务的预期解来通过测试。你的 agent 应当为给定任务生成走法或补丁。
- **绝不要暴露、记录或提交 API 密钥等凭据。** 请特别注意 `.env` 文件的内容。
- 作业期间你将使用多个 agent 及其多个版本。在任何时刻，只能为该 agent 使用规定的工具集合。

## 第 1 部分：构建编码 agent 并修复棋类应用

共享的 `Agent` 必须实现一个常规的 ReAct 循环：构建请求，获取 assistant 行动，执行其工具调用，添加关联的观察结果，并重复上述过程直到完成。

### 1. 构建提示

语言模型的提示是一系列消息，在交互的每一步都用来查询语言模型。消息是一个字典对象，至少包含 `role` 和 `content` 键，也可以包含额外信息。`content` 可以是字符串，也可以是包含推理 token 的结构化对象，或者（本作业未涉及的情况）图像。消息的 `role` 可以取若干值。

`system` 角色用于提供持续有效的指令——例如领域说明、环境信息、必须遵循的规则、适用于多个任务的一般策略等。`user` 角色提供任务信息[^1]，必要时也提供解决任务的补充指导。`assistant` 角色消息表示 LLM 的生成结果，`tool` 角色消息表示执行工具调用后得到的环境观察结果。你的任务是组织这些消息的顺序。按照约定，第一条消息必须且只能是 `system` 消息，并且在任何 assistant 消息之前要有一条 `user` 消息。由于 LLM 每次生成一个响应，一条 `assistant` 消息之后应当跟随 `tool` 消息（揭示工具调用结果）或 `user` 消息。关于消息结构的信息请参阅 [OpenAI API 参考](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create) 以及 vLLM 文档中的[示例](https://docs.vllm.ai/en/v0.7.2/getting_started/examples/openai_chat_completion_client_with_tools.html)。

你需要实现 `Agent.build_prompt` 方法。给定系统提示、作为 agent 类属性提供的任务提示，以及你可能添加的其他记录信息，该方法应准备发送给语言模型的输入。正如你所看到的，我们提供了 `Agent.query_language_model` 方法，它直接使用此方法的输出查询语言模型 API。你的实现需要为该查询提供正确的输入。`Agent.query_language_model` 方法会向语言模型 API 提供工具和推理参数，并返回包含所有相关信息的结构。它还维护步数，即 API 被查询的次数。API 会将提供的提示和工具渲染成一个 token 序列，这部分无需你处理。它还提供额外机制来记录请求以供评分，请不要修改这些机制。

agent 的 LLM 客户端会针对临时服务商故障重试，最多重试 `OPENAI_MAX_RETRIES` 次。

> **TODO(1.1.a)**
> 添加维护 agent 状态的机制，使其能够执行行动并观察结果。构造组成语言模型提示的一系列消息，其中应包括持续指令、任务说明、此前交互（包括观察结果）、此前轮次的推理和行动。注意，该方法应与领域无关，构造出的提示应适用于所有继承该类的领域特定 agent。

接下来，为 `CodeAgent` 构造系统提示和任务提示。系统消息**必须**逐字包含以下代码块，并使用环境提供的值：
```text
<system_information>
{
  "machine": <machine>,
  "release": <release>,
  "system": <system>,
  "version": <version>
}
</system_information>
```

> **TODO(1.1.b)**
> 构造 `CodeAgent` 的系统提示和 `task_prompt`。它们应当能够被 `Agent.build_prompt` 方法使用。

### 2. 运行 ReAct 循环

实现 `Agent.run` 的主体。

> **TODO(1.2)**
> 运行 ReAct 循环：协调提示语言模型生成推理和行动，提取模型生成的工具调用，并执行工具调用以获得 agent 下一步的观察结果。通过设置 `Agent.finished`，确保能够识别 agent 已完成任务的时刻。如果 agent 超过 `step_limit`，抛出 `StepLimitError`。

### 3. 执行编码工具

实现 `CodeAgent.execute_tool_calls`。编码 agent 支持两个工具：`execute` 和 `send_message`。工具定义位于[此处](./src/assignment/agent/tools.py)。你需要实现执行这些工具的机制，并准备显示给语言模型的输出。这些输出也是形如 `{"role": str, "tool_call_id": str, "content": str}` 的消息。每条消息都是执行一个工具后的观察结果，角色为特殊的 `tool`。[^2] 应使用 `Environment.execute` 方法，以确保代码在正确的 Modal sandbox 中执行。可以保留当前默认的可选参数，但必须允许 agent 在工具调用中覆盖这些参数。

> **TODO(1.3)**
> 让 agent 能使用 `execute` 和 `send_message` 工具。解析每个调用，执行已识别的工具，并为每个调用返回一条消息（一次 agent 响应中可能包含多个工具调用）。格式错误的 JSON 和未知工具必须转换为传回 agent 的、可恢复的观察结果，而不是抛出异常。

### 4. 加载技能

你可能以为现在已经可以让 agent 处理软件问题了，但还不行！`CodeAgent` 还需要知道如何提交解决方案。我们使用的具体协议是：agent 解决问题后，将解决方案保存到名为 `patch.txt` 的文件中。该文件会从环境中提取出来用于评估。

为了教会 agent 这一协议，我们将使用一个技能。技能是用于扩展 agent 能力的、专门且可复用的工作流。[Agent Skills 协议](https://agentskills.io/home)定义了一个标准，以便多个 agent 框架支持这些技能。`tasks/code-skills/submit-task` 定义了一个遵循该协议提交解决方案的基础技能。尽管该协议还有更多高级特性可供探索，但本作业只要求实现最小技能。

你需要实现的核心功能是_渐进式披露（progressive disclosure）_。为了让 agent 能使用许多技能，每个技能可能都很复杂，因此应让 agent 按需逐步获取信息。本作业中，你将实现一种简单的渐进式披露：在系统提示中提供所有可用技能的描述，同时向 agent 提供 `invoke_skill` 工具；agent 可以用技能名称调用该工具以查看完整技能。因此，当 agent 查看完整的 `submit-task` 技能时，它会知道如何提交补丁供评分。我们会检查系统提示是否提到了技能名称和描述。当 agent 没有可用技能时，提示的任何部分都不应提到 `patch.txt` 或提供提交说明。

技能文件位于[本地](./tasks/code-skills/)。如果 agent 使用技能运行，`Agent.skills_path` 会指向该位置（否则为 `None`）。在本作业中，agent 只会使用一个技能（`./tasks/code-skills/` 下的一个子文件夹），但原则上 agent 可能使用许多技能，因此渐进式披露会更加重要且实用。

> **TODO(1.4)**
> 验证 `skills_path`，发现每个子目录中的一个 `SKILL.md`，解析其 YAML frontmatter（文件开头两个 `---` 标记之间的内容），并返回一个以 frontmatter 中的 `name` 为键的映射。每个值必须包含用于模型技能目录的简洁 `metadata` 字符串，以及供 `invoke_skill` 使用的技能文件完整 `content`。如果名称重复，或 frontmatter 缺失/格式错误，应抛出清晰的 `ValueError`。如果 agent 有可用技能，应在提示中提供它们的描述/元数据。

例如，`SKILL.md` 的 `content` 为：
```
---
name: hello-world
description: Write "hello, world" to the terminal
---

echo "hello, world"
```
其 `metadata` 为：
```
name: hello-world
description: Write "hello, world" to the terminal
```

### 5. 运行并检查修复结果

现在让 agent 修复一个软件问题。

```bash
make run-code-agent
make check-part1
```
默认模型为 `deepseek-flash`。agent 会接收 `tasks/chess-terminal-move/problem_statement.md`，在 `/testbed` 中工作，并且必须复现、修复和验证失败。一次运行会生成：

- `artifacts/fix.patch`
- `artifacts/part1-trajectory.json`

`make check-part1` 会将生成的补丁应用到全新的 testbed，然后运行公共回归测试和棋类应用测试套件。不要直接编辑 `chess_app/` 中的目标代码。

完成第 1 部分后，你的解决方案应通过与提示构造、截断、步数限制、格式错误/未知工具和补丁提交相关的测试。

## 第 2 部分：实现上下文压缩

较长的 ReAct 对话记录会增加成本，并最终挤占有用的上下文。随着 agent 执行越来越复杂、跨度越来越长的任务，它们也会达到语言模型上下文窗口的限制。为了让 agent 更高效地处理运行时间更长的任务，我们希望只保留此前行动和观察中相关的信息，并将其压缩为工作记忆。现在，你将在共享的 `Agent` 中实现由模型生成的工作记忆；不要使用服务商特定的上下文压缩端点。

> **TODO(2.1)**
> 实现 `Agent.compact_context`。提示模型压缩上下文。压缩系统提示应要求生成简洁、事实性的工作记忆，并保留目标、约束、文件、命令、编辑、具体结果、失败方法、测试、阻塞因素和下一步行动。只总结较早的前缀；原始系统/任务消息必须逐字保留，并至少保留最近一条完整的 assistant 行动及其关联的全部工具观察结果。压缩后的摘要应改变 `build_prompt` 的输出，并减少提示长度。

**不要**修改 agent 类的 `api_prompt` 和 `api_responses` 属性。这些属性用于记录和评估。

> **TODO(2.2)**
> 在共享循环每次请求新的行动之前调用 `maybe_compact_context()`。该方法已经会估计活动 token 数并处理阈值，同时跟踪压缩事件以供记录。

使用 6,000 token 阈值运行 vendored 的 `django__django-15368` 任务：

```bash
COMPACT_THRESHOLD=6000 \
SWEBENCH_PATCH=artifacts/django__django-15368.patch \
SWEBENCH_TRAJECTORY=artifacts/django__django-15368-trajectory.json \
make run-swebench-agent INSTANCE=django__django-15368
make check-swebench INSTANCE=django__django-15368
```

将 `COMPACT_THRESHOLD=0` 设置为不启用压缩标志，并运行完整上下文基线。保留两次运行时使用不同的输出名称，例如：

```bash
COMPACT_THRESHOLD=0 \
SWEBENCH_PATCH=artifacts/django__django-15368-baseline.patch \
SWEBENCH_TRAJECTORY=artifacts/django__django-15368-baseline-trajectory.json \
make run-swebench-agent INSTANCE=django__django-15368
```

提交的压缩运行必须至少触发一次压缩，显著减少活动上下文，并生成能通过 `check-swebench` 的补丁。生成过程具有随机性，因此它不必比每一次基线样本都使用更少的 ReAct 步数。

生成压缩和不压缩两种轨迹后，比较两种轨迹的 token 用量。在 `artifacts/token-usage-analysis.md` 中记录观察结果，并简要解释出现这些趋势的原因。压缩和不压缩条件下的上下文用量有哪些权衡？

## 第 3 部分：构建 `ChessAgent`

你的 agent 已经在第 1 部分修复了棋类应用的问题，现在可以构建一个下棋 agent。`ChessAgent` 复用第 1、2 部分的完整循环。它执白，服务器上的确定性 bot 执黑。每次白方走出合法一步后，服务器会自动应答。

### 1. 实现 `play_move`

首先定义一个新工具，让 agent 能够下正在应用中运行的棋局。按照 [OpenAI 函数调用指南](https://developers.openai.com/api/docs/guides/function-calling)定义工具。

> **TODO(3.1.a)**
> 定义名为 `play_move` 的 OpenAI function-tool schema。它必须恰好接受一个名为 `move` 的必需字符串参数，并说明走法使用 [UCI 表示法](https://en.wikipedia.org/wiki/Universal_Chess_Interface)（例如 `e2e4` 和 `e7e8q`），同时拒绝额外参数。

接下来，实现运行工具的机制。[^3]

> **TODO(3.1.b)**
> 在 `chess_tools.py` 中实现 `_play_move`。解析参数，并向 `/api/move` POST `{"move": <uci move>`。返回序列化后的 JSON 对象。捕获工具抛出的所有错误，并返回包在 `<chess_error></chess_error>` 中的错误消息，供 agent 处理。覆盖格式错误的 JSON 参数、参数不是对象、缺少 `fen` 或 `fen` 不是字符串、`move` 不是字符串、服务器拒绝某个局面或走法，以及传输失败。

你可以查看 `chess_app` 了解 API 的工作方式。

使用 `format_state` 方法格式化成功返回的状态，更新 `last_state`，并根据 `game_over` 设置 `finished`。使用原始的 `tool_call_id` 关联观察结果。格式错误的 JSON、未知工具、非法走法和网络错误都应成为 `<chess_error>...</chess_error>` 观察结果。由于走法后实时局面会变化，一组并行调用中最多执行一个走法，其余调用应以可恢复的方式拒绝。

初始观察结果和成功走子后的观察结果已经显示棋盘、黑方应答和下一步合法走法。不需要单独的读盘工具。

运行：

```bash
make run-chess-agent
```

该命令会应用你的 `fix.patch`，启动修复后的服务器，并保存：

- `artifacts/part3-trajectory.json`
- `artifacts/game-result.json`

输出的 HTTPS URL 提供棋盘和 API。agent 下棋时，棋盘会轮询实时状态；棋盘上的按钮只刷新状态，不会重置棋局。
`CHESS_TIMEOUT=1800` 控制 sandbox 的生命周期。

### 2. 运行观察结果 A/B 实验

为了了解工具接口对 agent 行为的影响，你将针对两个模型比较“仅棋盘观察”和“棋盘加合法走法”两种情况。运行器会配置这些选项，无需修改源码，并使用不同的文件名：

```bash
make run-obs-deepseek-no-legal
make run-obs-deepseek-legal
make run-obs-gpt-oss-no-legal
make run-obs-gpt-oss-legal
```

对于四次运行，记录 `play_move` 总调用次数、被判定为非法的调用次数、无效走法比例，以及是否达到 `game_over: true`。在 `artifacts/observation-experiment.md` 中写一份简短比较。评分依据是实验过程和证据，而不是某个特定结果或是否赢棋。

### 3. 添加 `simulate_move`

为了让 agent 能够规划更复杂的策略，你将赋予它模拟走法的能力。模拟可以计算走出某一步后的效果，但不会实际改变真实棋盘状态。你将定义并注册 `SIMULATE_MOVE_TOOL`，实现 `_simulate_move`，并将其加入现有分发器。

> **TODO(3.3.a)**
> 像 `play_move` 工具一样定义 `simulate_move` 工具。

`simulate_move(fen, move=None)` 会向 `/api/simulate` 发送 `POST` 请求。

仅传入完整的六字段 [FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) 给 `simulate_move`，会返回该局面及其合法走法。传入 FEN 和 UCI 走法，则返回恰好走出一个 ply 后的局面，白黑双方均适用。返回 JSON，以便 Python 使用其中的 `fen`、`squares`、`turn`、`legal_moves` 和终局结果字段。

> **TODO(3.3.b)**
> 在 `chess_tools.py` 中实现 `_simulate_move`。解析参数，使用提供的 `/api/simulate` 端点和 FEN 及可选走法发起调用，并返回序列化后的 JSON。捕获工具抛出的所有错误，并返回包在 `<chess_error></chess_error>` 中、供 agent 处理的错误消息。覆盖格式错误的 JSON 参数、参数不是对象、缺少 `fen` 或 `fen` 不是字符串、`move` 不是字符串、服务器拒绝某个局面或走法，以及传输失败。

请求应使用 `ChessAgent.chess_client` 发出。

### 4. 添加 `run_python`

有了在棋盘上模拟走法的能力，agent 就能进行更复杂的规划。执行代码可以让 agent 更可靠地执行自己提出的计划，你将通过程序化工具调用启用此功能。定义并注册 `RUN_PYTHON_TOOL`，实现 `_run_python`，并将其加入分发器。模型提供 `run_python(code)`，代码片段可以像普通同步 Python 函数一样调用 `simulate_move` 和 `play_move`。

> **TODO(3.4)**
> 解析参数，并在 sandbox 中运行代码，同时让已注册的工具按名称可用。`/opt/assignment/sandbox_python.py` 是 `env` sandbox 中的脚本，它可以访问本文件中的相同工具定义。使用该脚本运行作为 `run_python` 工具参数、由模型生成的代码。脚本接受两个位置参数——`port` 和 base64 编码的代码字符串（用于避免引号问题）。实现此工具调用。
> 脚本会打印一个包含 `stdout`、`stderr` 和 `error` 的 JSON 对象；原样返回该字符串。返回码非零表示 sandbox 本身失败，而不是模型代码失败。将 `exception_info` 或 `stderr` 报告为 `<chess_error>`。
> 如果出现类型不匹配或解析失败等问题，返回 `<chess_error>{message}</chess_error>`。

模型生成的代码不能在本地 agent 进程中运行。`_run_python` 应通过 `env.execute` 将 base64 编码的代码发送到提供的 sandbox runner：

```text
python /opt/assignment/sandbox_python.py <port> <base64-code>
```

返回 runner 的 JSON 字符串，其中包含 `stdout`、`stderr` 和 `error`。sandbox 命令的非零返回码应转换为 `<chess_error>`。代码片段抛出的 Python 异常表示 runner 调用成功，应放在其 `error` 字段中。每个代码片段执行后，重新读取实时棋盘，更新 `last_state` 和 `finished`，并将格式化后的状态追加到观察结果中；否则模型可能重复执行代码片段已经提交的走法。

使用以下命令启用这些工具：

```bash
uv run assignment-play-chess --programmatic-tools
```

实现完成后，你的 agent 应能生成一段代码来选择走法，并在棋局中执行该走法。你可能仍会发现，下棋 agent 不会充分使用手头的工具来下出好棋，而是经常直接调用 `play_move` 行动。为了给它更明确的结构和策略，我们再次使用之前探索过的技能概念。

### 5. 加载并使用棋类技能

虽然执行 Python 代码让 agent 可以选择执行复杂计划，但 agent 未必会主动这样做。为了给它一个具体的执行策略，我们在 `tasks/chess-skills/select-move` 中提供了另一个技能。你将赋予 `ChessAgent` 一部分 `CodeAgent` 的技能使用能力。由于工具执行机制略有不同，你必须在 `ChessAgent` 中重新实现 `invoke_skill` 工具的处理逻辑 `_invoke_skill(skills, arguments)`，仅当加载了技能时注册 `INVOKE_SKILL_TOOL`，并返回指定技能的完整内容。

> **TODO(3.5)**
> 解析参数并返回指定技能的内容。如果出现类型不匹配或解析失败等问题，返回 `<chess_error>{message}</chess_error>`。

```bash
uv run assignment-play-chess \
  --programmatic-tools \
  --skills-path tasks/chess-skills \
  --trajectory artifacts/part3-python-skill-trajectory.json
```

轨迹必须显示先调用 `invoke_skill`，随后执行调用 `simulate_move` 进行搜索的 `run_python` 代码，再调用一次 `play_move` 提交走法。只读取技能、然后每回合直接调用一次 `play_move`，不能证明这些工具协同工作。棋局不必结束，也不必获胜。

## 评分

本作业满分 **100 分**。每行独立评分；一次随机性模型运行失败，不会抹掉其他实现部分的得分。

| 部分 | 标准 | 分值 | 证据 |
|---|---|---|---|
| 1 | 提示构造以及有效的行动/观察历史 | 6 | 私有单元测试 |
| 1 | ReAct 生命周期、纯文本恢复、步数限制、清理、轨迹 | 6 | 私有单元测试 |
| 1 | 编码工具分发、可恢复错误、补丁提交 | 6 | 私有单元测试 |
| 1 | 棋类补丁应用并通过私有/回归测试 | 8 | 在全新 testbed 中重放补丁 |
| 1 | 技能发现和 `invoke_skill` 行为 | 4 | 私有测试 |
| 2 | 压缩触发和模型生成的摘要 | 6 | 私有测试和轨迹 |
| 2 | 原始指令及最近完整工具步骤仍然有效 | 6 | 私有测试和轨迹 |
| 2 | 可审计的压缩，显著减少活动上下文 | 4 | 压缩事件和用量 |
| 2 | 完整上下文与压缩 token 用量分析报告 | 4 | 报告 |
| 2 | SWE-bench 补丁通过 FAIL_TO_PASS 和 PASS_TO_PASS | 8 | 重放补丁 |
| 3 | `play_move` schema 和注册 | 4 | 私有测试 |
| 3 | `play_move` 状态更新和可恢复错误 | 6 | 私有测试 |
| 3 | 基础棋局轨迹达到终局 | 4 | 轨迹和结果 |
| 3 | 四次运行的观察结果 A/B 实验完整 | 4 | 四条轨迹、四个结果 |
| 3 | 观察结果 A/B 实验报告 | 4 | 报告 |
| 3 | `simulate_move` 无状态行为和错误处理 | 6 | 私有单元/集成测试 |
| 3 | `run_python` sandbox 执行、状态刷新、错误处理 | 6 | 使用 sandbox 替身的私有测试 |
| 3 | 技能轨迹结合技能、程序化搜索和实时走法 | 6 | 轨迹重放 |
| — | 完整、可解析且符合规则的提交 | 2 | 归档验证 |
| | **总计** | **100** | |

评分器会重放补丁和提交的轨迹，不会发起新的 LLM 调用。缺少或不一致的证据只会影响对应行的得分。教师测试和参考补丁不包含在本仓库中。

## 提交

提交一个 ZIP 归档，其中包含你在 `src/assignment/agent/` 下修改的文件，以及以下产物：

```text
artifacts/fix.patch
artifacts/part1-trajectory.json
artifacts/django__django-15368.patch
artifacts/django__django-15368-trajectory.json
artifacts/token-usage-analysis.md
artifacts/part3-trajectory.json
artifacts/game-result.json
artifacts/part3-no-legal-moves-deepseek.json
artifacts/part3-no-legal-moves-deepseek-result.json
artifacts/part3-legal-moves-deepseek.json
artifacts/part3-legal-moves-deepseek-result.json
artifacts/part3-no-legal-moves-gpt-oss.json
artifacts/part3-no-legal-moves-gpt-oss-result.json
artifacts/part3-legal-moves-gpt-oss.json
artifacts/part3-legal-moves-gpt-oss-result.json
artifacts/observation-experiment.md
artifacts/part3-python-skill-trajectory.json
```

只有在修改过时才加入 `src/assignment/prompts.py`。不要提交凭据、`.env`、任务文件、测试、子模块内容或教师文件。你的 ZIP 应包含 `src/` 和 `artifacts` 子文件夹，以及 `AI_USAGE.md` 文件。`AI_USAGE.md` 应详细说明你在本作业中使用 AI 技术的情况。列出使用过的所有工具，并清楚描述每个工具的用途。如果没有使用任何 AI 辅助，也请在该文件中声明。我们不会为提交的 `AI_USAGE.md` 文件评分，但会通过测验检查你对所提交代码的理解（具体安排将另行通知）。

[^1]: 在离线评估环境中，通常会将任务组织为一条来自用户的请求。更多示例请参阅这些[文档](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/chatgpt?tabs=python-key%2Cdotnet-secure%2Cjavascript-secure&pivots=programming-language-python)。

[^2]: 服务商返回重复的 `call_0` ID 是合法的：将每条工具观察结果与同一 assistant 行动中的调用匹配，不要假设 ID 在整个轨迹中全局唯一。

[^3]: `_play_move` 和其他工具被隔离在 `chess_tools.py` 中，以便能够在远程 sandbox 中执行这些工具。请遵循现有代码结构，正确使用 Modal sandbox 执行棋步。
