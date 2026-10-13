# 11-768 作业 2：评估数据可视化 Agent

## 概述

在将 agent 部署到真实环境之前，可靠的评估对于判断其能力和稳定性至关重要。然而，与非 agent 模型评估相比，agent 系统的复杂性以及任务持续时间较长，会带来许多新的问题。在本作业中，你将亲自设计验证器，对 agent 轨迹进行评估，从而更好地理解评估开发中的挑战。

本作业围绕**数据可视化 agent**展开：这类 LLM agent 接收数据和用户指定的要求，并为用户生成数据可视化结果。首先，你将实现一个验证器，为数据可视化 agent 的输出评分。给定用户任务、agent 轨迹和最终图像，你需要判断 agent 是否正确，或者诊断运行过程中的失败。随后，你将设计额外任务，探查数据可视化 agent 的能力边界和弱点，并改进已有评估。最终目标是设计一个能够泛化到一组私有 agent 轨迹的验证器，课程将基于这组私有轨迹评分。在评分时，验证器每次只评估一个运行结果；它不会收到真实标签、参考图像或其他候选运行结果，只能看到轨迹和最终图像。

## 初始设置

在学生发布目录中安装环境：

```bash
uv sync
```

开发期间，你将在自己的 Modal workspace 中使用课程提供的 Modal 额度运行验证器模型。先认证 Modal CLI，再部署验证器端点：

```bash
uv run modal setup
scripts/deploy_validator_model.sh
```

部署命令会打印端点 URL。为你的 Modal workspace 创建 proxy token，然后配置 OpenAI 兼容客户端：

```bash
cp .env.example .env
# 将端点 URL（包括 /v1 后缀）和 proxy token 填入 .env。
```

调用端点时才需要 proxy token，因此可以先完成 Modal 设置和部署，再创建 token。不要提交 `.env`、proxy token 或其他凭据。本作业使用的验证器模型是 **Qwen/Qwen3-VL-30B-A3B-Instruct-FP8**；不得替换该模型，也不得让验证器调用其他模型。因此，你设计的验证器应具有泛化能力，而不是依赖更强的裁判模型。评分时，课程会使用同一个模型的部署来调用你的验证器。如果需要构建新的镜像，GPU 容器加载模型大约需要 5–10 分钟。

### 评估格式

你需要在 `validator/solution.py` 中实现 `validate(run)`。对于每个运行结果，它会收到任务、输入数据、agent 轨迹和最终图像，并返回错误列表。空列表表示该运行结果可以接受。每个错误属于以下四个固定类别之一，并且需要提供基于运行结果的具体证据：

| 类别 | 含义 |
| --- | --- |
| `execution_failure` | agent 最终没有生成有效图像。 |
| `wrong_data` | 绘制的数据不符合任务要求。 |
| `wrong_chart` | 图像没有遵循要求的图表设计。 |
| `hard_to_read` | 生成的图像难以阅读或无法阅读。 |

在真实标签中，`execution_failure` 是终止性类别：当人工标签表明运行结果没有有效图像时，该运行不会计入其他三个类别的 TP、TN、FP 或 FN。验证器预测出 `execution_failure` 并不会使它免于判断其他类别。
`validator/prediction.py` 定义了准确的输出格式，`validator/baseline.py` 提供了完整示例。

每个类别都在所有相关运行结果上作为一个二分类问题单独评分。如果真实标签列出了某个类别，则该运行在该类别上为正例；如果验证器返回的错误列表包含该类别，则预测在该类别上为正例。一个运行可以同时包含多个错误，因此会在每个列出的类别中都作为正例。统计评估中的真正例（TP）、真负例（TN）、假正例（FP）和假负例（FN)，并计算 Matthews 相关系数（MCC）：

$$
\mathrm{MCC}=\frac{TP\cdot TN-FP\cdot FN}
{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}.
$$

MCC=1 表示完美分类，MCC=0 表示没有相关性，负值表示反向相关。分数范围为 [-1, 1]。总体 `macro_mcc` 是四个类别 MCC 的非加权平均值，分母始终为四。

当真实标签同时包含正类和负类、但预测是常量时，MCC 为 0。当某一类别的真实标签缺少正类或负类时，评分器按约定报告 MCC=0，同时标记支持不足并给出正类和负类的支持数量。小规模的自建数据集很容易出现这种情况：某类别可能从未出现，也可能在每个运行中都出现。应谨慎解释这类类别的分数。私有评分集的每个类别都包含两种标签，因此这一约定不会影响正式成绩。MCC 只衡量分类结果；验证器解释中证据的质量会单独评估。

### 各类错误的定义

四个类别都是二元判断，一个运行可以同时带有多个类别。下面是种子集标签和隐藏评估集标签中各类别的完整定义。

**`execution_failure`。** agent 最终没有生成有效图像。该类别始终单独出现。

**`wrong_data`。** 图中绘制的内容不是指令要求的数据：

- *数值错误。* 绘制的数值、位置或派生量与要求的数据或公式不同：聚合、缩放或归一化错误；函数或字段使用了错误的范围或网格采样；本应编码数据的气泡大小、颜色值或误差条编码错误。
- *数据缺失。* 要求的序列、类别、分组、面板数据或数据点没有绘制，或者丢失了某个维度（例如要求绘制三维量，却少绘制了一个变量）。
- *选择错误。* 使用了错误的子集、列或筛选条件，或者混入了未要求的序列。
- *顺序错误。* 点、柱、类别或序列的顺序与要求不同：要求排序时未排序，序列被反转或打乱，或者面板没有展示分配给它的数据。

**`wrong_chart`。** 数据本身正确，但没有遵循要求的图表设计。只要任务要求了以下内容，缺失或错误都计入该类别：

- *图表类型和投影。* 图表类型错误（例如要求柱状图却使用折线图、缺少饼图）；要求堆叠却绘制分组柱；要求水平柱却绘制垂直柱；要求 3D 却绘制成 2D；缺少极坐标投影、等高线或曲面；双坐标轴缺失或在禁止时使用；缺少缩放插图或其连接线。
- *布局。* 子图数量或排列错误，图像尺寸或 DPI 错误，面板没有按要求共享坐标轴。
- *坐标轴。* 缺少或错误的 x、y、z 轴标签；标题或标签文本与要求不完全一致；坐标轴范围错误（没有从零开始、范围不正确或裁切了要求的数据）；坐标轴尺度错误（线性尺度代替对数尺度、等高线级别不是对数级别）；要求隐藏时仍显示刻度或刻度标签；刻度标签错误或旋转角度错误；要求隐藏时仍显示坐标轴脊；缺少网格。
- *图例、色条和标注。* 缺少图例；缺少色条或色条与数据不匹配；缺少要求的标注或标记点，或位置错误；缺少要求的函数标签或柱值标签；要求的公式没有渲染；阴影区域错误。
- *样式。* 使用错误或被禁止的颜色映射（例如要求感知均匀色图却使用非均匀色图）；线型错误（实线代替虚线或点线）；标记形状错误；命名序列的颜色错误；要求填充却未填充；缺少箱须、中位数标记、连接线或箱体；多边形没有闭合；没有应用要求的线偏移。
- 如果要求的文本元素（标题、坐标轴标签）完全落在保存图像之外，也计为缺失，因此属于 `wrong_chart`。

**`hard_to_read`。** 图像整体按要求绘制，但读者无法阅读，或只能费力阅读：

- *文字被裁切。* 标题、坐标轴标签、刻度标签或标注被图像边缘部分裁切（标签被切掉一部分属于 `hard_to_read`；完全位于图像之外则属于 `wrong_chart`）。
- *文字重叠。* 刻度标签、标注、标题或其他标签相互重叠，或者标记覆盖了标注它的文字。
- *元素遮挡内容。* 图例、标注或文本框遮挡数据、标签或其他文字。图例位于空白区域时不算错误。
- *低对比度或不可见内容。* 文字或数值标签与背景的对比度低于约 2.5:1；序列使用背景色绘制、线宽为零，或因其他原因不可见。
- *数据无法区分。* 必须区分的两个序列或分组使用相同或非常接近的颜色、线型和标记；颜色映射使本应不同的数值看起来无法区分。
- *布局拥挤。* 数据、箱体或面板因坐标轴范围或坐标轴重叠而被压缩成难以阅读的狭窄区域。

## 第 1 部分：检查种子 agent 运行结果

我们在 `artifacts/<run_id>/` 下提供了一些种子 agent 运行结果。每个运行子目录都包含 `result.json`（其中包括任务说明、完整 agent 轨迹和 agent 运行结果），以及 `figure.png`（如果 agent 成功生成图像）。`run_id` 是唯一标识；多个运行可以共享同一个 `task_id`，因此索引时应使用 `run_id`。我们还提供了一个简单的、基于 VLM 的验证器，位于 `validator/baseline.py`。可以对所有已有运行结果运行基线验证器：

```bash
uv run python -m validator.runner \
  --solution validator.baseline \
  --artifacts artifacts \
  --out output/baseline_predictions.json
```

如果模型拒绝请求并指出上下文窗口超限，基线验证器会发出警告，然后使用轨迹和输入文件的较短摘录重试一次，保留它们的开头和结尾。回退方案使用服务端 tokenizer 和返回的上下文限制，将文本和响应预算设为上下文的 50%，其余空间留给图像和格式化内容。`validator/baseline.py` 中的 `CONTEXT_FRACTION` 控制这个比例；不存在固定的上下文大小，也不会逐步缩小并反复重试。任务和图像保持完整，省略内容会在提示中明确标记。这个回退方案可能丢失相关证据，因此判断结果不保证与完整请求相同。如果压缩后的请求仍然超出限制，或者服务端 tokenizer 端点不可用，则会继续抛出错误。这只是基线验证器的做法；你可以在 `validator/solution.py` 中自定义上下文管理。

我们还提供了种子 agent 运行结果的人工真实标签，位于 `seed_labels.json`。使用以下命令对基线预测评分：

```bash
uv run python -m workflow score \
  --labels seed_labels.json \
  --predictions output/baseline_predictions.json
```

首先比较真实标签和验证器标签。报告四个类别各自的 MCC 分数以及 macro-MCC，然后找出两类常见的验证器错误；每类错误都必须出现在至少两个 agent 轨迹中，并列出所有相关轨迹的运行 ID。对于每类错误，描述可观察到的证据，并提出验证器可以采用的改动来捕获这类错误。

## 第 2 部分：改进验证器

根据第 1 部分的分析，实现对默认验证器提出的两项改进。记录你所做的修改、改进后验证器在种子运行上的表现，并与基线验证器进行比较。即使性能没有显著提升也没有关系，后续还会继续改进。报告改进后验证器四个类别各自的 MCC 分数和 macro-MCC，并定性说明修改是否确实缓解了你发现的错误类型。具体来说，至少指出一个假正例和一个假负例，并尝试解释这些错误的原因，以及它们说明验证器可能缺少什么能力。

在 `validator/solution.py` 中实现修改，然后在发布的种子运行上运行修改后的验证器：

```bash
uv run python -m validator.runner \
  --artifacts artifacts \
  --out output/pre_iteration_seed_predictions.json
uv run python -m workflow score \
  --labels seed_labels.json \
  --predictions output/pre_iteration_seed_predictions.json
```

验证器模型是固定的：不得替换验证器实际使用的模型，也不得调用其他模型。因此，应尽量设计与模型无关的验证器。除此之外，只要 `validate(run)` 函数签名和 `validator/prediction.py` 中的输出格式保持不变，你可以任意定制 `validator/solution.py`。需要使用输入数据、最终输出和 agent 轨迹进行评估；基线验证器展示了如何将三者都发送给模型。

## 第 3 部分：设计新的评估任务

现在，你应该已经基于第 1 部分的定性观察改进了验证器。通过阅读真实 agent 轨迹识别潜在错误，是改进验证器的一种常见方式；另一种方式是使用分布外数据测试验证器的稳健性。

首先阅读当前的学生验证器实现，找出它可能难以泛化的两类可视化任务。编写五个可视化任务（每类任务至少两个），运行提供的 agent，并根据上面四种错误类别为结果打标签。如果可视化任务需要自定义数据，可以使用 LLM 合成数据，也可以复用其他可视化任务的数据。随后在这些对抗性案例上运行验证器，并报告四个类别各自的 MCC 分数以及 macro-MCC。不得对公开任务 ID 或标签进行硬编码。

提供的 agent 模型是固定的：

- **Qwen/Qwen2.5-Coder-3B-Instruct**
- **mistralai/Ministral-3-14B-Instruct-2512**
- **zai-org/GLM-4.7-Flash**

每个新任务都要使用每个提供的 agent 运行一次，总计至少生成 15 个自建 agent 运行结果。人工审核每个生成的运行结果，并使用相同的证据标准。在报告中分析哪些失败反映了可视化 agent 的弱点，哪些失败反映了验证器的弱点。

### 开始第 3 部分之前

```bash
scripts/deploy_generation_models.sh
# 将打印出的三个 URL 填入 .env 中对应的 GENERATION_* 设置，
# 给每个 URL 追加 /v1，并设置 GENERATION_API_KEY。
scripts/deploy_agent_runner.sh
```

注意：每个 agent 最多执行 20 个步骤；每条 shell 命令最多运行 120 秒；总墙钟时间最多 15 分钟。部署故障排查见 `infrastructure/README.md`。

### 编写任务

在 `tasks/<task_id>/` 下为每个任务创建一个目录：

```text
tasks/<task_id>/
  task.json
  inputs/                 # 可选；任务专属输入文件
```

其 `task.json` 描述文件必须正好包含以下字段：

```text
{
  "task_id": "sales-by-region",
  "task_class": "multi-series aggregation",
  "instructions": "Using matplotlib, read sales.csv and ... Save figure.png.",
  "inputs": [
    {"name": "sales.csv", "path": "inputs/sales.csv"}
  ]
}
```

`task_id` 必须与目录名一致，并且只能包含字母、数字、句点、下划线和连字符。`task_class` 用于命名你识别出的两类弱点之一；同一类任务应使用相同的值。`instructions` 必须完整指定图表，并且应使人类能够审核其正确性。每个输入的 `name` 是 agent 看到的普通文件名；`path` 是相对于任务目录的路径。路径不能是绝对路径，也不能包含 `..`。当提示中包含全部数据时，使用空的 `inputs` 列表。`artifacts/` 下发布的运行结果提供了任务说明和输入文件的示例。你的任务不能依赖现有环境中未包含的额外包（现有环境包括 matplotlib、numpy、pandas、Pillow、scipy 和 seaborn）；私有评估集也不会要求你考虑需要其他包的任务。

在生成运行结果前，验证所有描述文件和引用的文件：

```bash
uv run python -m workflow validate-tasks
uv run python -m workflow generate
```

运行结果支持断点续跑：已有任务/模型运行会被跳过。要重新生成已有运行结果，添加 `--force`；开发期间可以使用 `--models qwen`、`--models ministral` 或 `--models glm` 来运行子集。

每个生成的运行结果都写入以下目录结构：

```text
runs/<task_id>__<agent>/
  result.json             # 任务、agent 身份和完整轨迹
  inputs/                 # 该运行使用的任务输入副本
  figure.png              # agent 留下输出文件时存在
```

在生成的运行结果上运行验证器：

```bash
uv run python -m validator.runner \
  --artifacts runs \
  --out output/pre_iteration_authored_predictions.json
```

### 人工标签和评分

创建 `labels.json`，为每个生成的运行结果添加一个空白条目：

```bash
uv run python -m workflow init-labels
```

然后人工审核每个运行结果。对于每个条目，如果运行结果可以接受，将 `"errors": null` 替换为 `[]`；否则替换为由 `{family, evidence}` 错误对象组成的列表。每个错误都必须提供具体证据，并且 `execution_failure` 必须单独出现。示例见 `labels.example.json`。

验证覆盖范围和格式，然后报告四个类别的 MCC 分数和 macro-MCC：

```bash
uv run python -m workflow validate-labels
uv run python -m workflow score \
  --predictions output/pre_iteration_authored_predictions.json
```

### 将任务打包为 Harbor 环境

[Harbor](https://github.com/harbor-framework/harbor) 是一种评估框架，近年来已成为 agent 基准测试的行业标准，并因 Terminal-Bench 2.0 而得到推广。本部分将把当前只支持内部 `workflow generate` 的任务打包为 Harbor 格式，使任何人都能运行这些任务。由于 Harbor 任务要求确定性的验证器，你可能需要为每个私有任务增加功能，但不要重新设计任务的实际内容。具体来说，你必须要求 agent 写入 `plot.py` 和 `plotted_values.json`，而原始任务没有要求这两个文件。任何此类追加要求都应放在 `harbor/tasks/<task_id>/` 中。更多信息见 `harbor/README.md`。

先查看 `harbor/example/` 中的合成示例，了解具体要求。你应该分别使用示例的参考解决方案和空操作 agent 运行它：

```bash
uv run harbor run -p harbor/example -a oracle --job-name example-oracle   # reward 1.0
uv run harbor run -p harbor/example -a nop    --job-name example-nop      # reward 0.0
```

随后，至少运行 `harbor/example/mutants/` 中四个故意错误的示例解决方案之一。三个方案得分为 0.0；`illegible/` 得分为 1.0，因为验证器漏检了其图像太小而难以阅读的问题。如何替换 mutant 并重新运行 Harbor 验证器，见 `harbor/example/README.md`。之后打包你自己的每个任务。你可以使用以下命令，根据已经编写的私有任务初始化 Harbor 任务（它会将 Harbor 模板复制到每个已有任务对应的 `harbor/tasks/<task_id>/` 目录，然后填充一些样板信息和初始 `instruction.md`）。**生成运行结果后不要编辑种子提示。**

```bash
uv run python -m workflow package
```

对于每个私有任务，你需要在 `harbor/tasks/<task_id>/tests/test_state.py` 中实现确定性检查，在 `harbor/tasks/<task_id>/solution/solve.sh` 中实现参考解决方案，并完成其他标记为“Required Outputs”的内容。不要在任务上消耗 Docker 周期之前检查它，然后确认参考解决方案确实通过：

```bash
uv run python harbor/preflight.py harbor/tasks/<task_id>
uv run harbor run -p harbor/tasks/<task_id> -a oracle --job-name <task_id>-oracle
```

要完成本节，所有打包任务都必须让参考解决方案获得 1.0。除此之外，在所有打包任务中总共编写两个 mutant 方案，路径为 `harbor/tasks/<task_id>/mutants/<name>/solve.sh`。两个方案都应产生合理但错误的答案；区别在于验证器对它们的处理。一个必须得分为 0：你的检查应拒绝它。最简单的做法是利用已经要求生成的 `plotted_values.json`，手工推导期望数字，将其作为字面量写入测试文件进行比较；任何修改数值的 mutant 都会失败。另一个必须得分为 1.0：你的检查应放过一个真正错误的答案。这个 mutant 应暴露验证器的真实局限。例如，图像可以包含正确的数据和标签，却由于在小画布上使用 2pt 文字而难以阅读；示例中的 `harbor/example/mutants/illegible/` 展示了一个对讲义中最强验证器得分为 1.0 的实现。`check-submission` 会运行所有这些检查，因此不能只在报告中描述而不实际运行。在报告中，给出每个 mutant 的得分、它哪里出错，以及捕获它的断言或为什么没有断言能够捕获它。第二个 mutant 正是本部分的重点：它用于测量，而不是直接告诉你，VLM 验证器需要跨越的边界在哪里。`harbor/TROUBLESHOOTING.md` 涵盖了最可能遇到的故障，包括无效 `task.toml` 被报告为“no task found”的问题。

## 第 4 部分：迭代改进设计

根据第 3 部分的结果，至少再对验证器做一次改进。在种子任务和新任务上评估最终验证器，并报告四个类别各自的新 MCC 分数和 macro-MCC，同时定性分析这次修改是否改善了手写任务上的表现。和前面一样，至少指出一个假正例和一个假负例，包含轨迹中的相关文本，并提出验证器判断错误的假设原因。

在两组轨迹上运行最终验证器：

```bash
uv run python -m validator.runner \
  --artifacts artifacts \
  --out output/final_seed_predictions.json
uv run python -m validator.runner \
  --artifacts runs \
  --out output/authored_predictions.json
uv run python -m workflow score \
  --labels seed_labels.json \
  --predictions output/final_seed_predictions.json
uv run python -m workflow score \
  --predictions output/authored_predictions.json
```

完成上述步骤后仍可以继续改进验证器或设计任务，因为最终验证器在私有测试集上的表现会占一部分成绩。请在报告中报告对验证器做出的所有额外修改，说明采用每项修改的动机，并定性和定量地分析其效果。

## 提交与评分

提交以下内容：

```text
validator/solution.py   # 最终验证器实现
output/                 # 种子运行和自建运行的预测结果
tasks/                  # 五个或更多自建任务及其输入
runs/                   # 每个自建任务的三个固定 agent 运行结果
labels.json             # 每个自建 agent 运行结果的一份人工审核标签
harbor/tasks/           # 每个自建任务打包后的 Harbor 环境，包含两个 mutant 方案
report.pdf              # 编译后的报告
report.tex              # 报告源文件
AI_USAGE.md             # 声明使用过的 AI 工具（不评分）
```

提交前运行端到端结构检查：

```bash
uv run python -m workflow check-submission
```

它会验证必需文件，验证每个任务和输入，要求至少五个任务、至少两个 `task_class`，且每类至少两个任务；要求每个任务恰好有三个固定 agent 的一个运行结果，并检查 `labels.json` 是否恰好覆盖这些运行结果。它还会分两个阶段检查 Harbor 打包结果。结构检查包括：每个自建任务对应一个 `harbor/tasks/<task_id>/` 且没有多余目录；每个任务都包含完整的 `harbor/template/` 布局；参考解决方案和验证器中不再有模板哨兵；`instruction.md` 中没有 TODO 标记；`instruction.md` 仍然逐字包含描述文件中的提示，并要求生成 `plot.py` 和 `plotted_values.json`；所有任务下总共恰好有两个 `mutants/<name>/solve.sh`。它会一次报告所有任务中的所有问题。随后会运行 Harbor，并验证奖励约定：每个打包任务使用 `-a oracle` 必须得到 1.0；一个 mutant 必须得 0；另一个 mutant 必须得 1.0。它会打印读取到的奖励；如果失败，会指出对应的 `jobs/check-submission-<job>/<trial>/verifier/pytest.log`，其中包含解释。该阶段需要 Docker，耗时几分钟；`--skip-harbor-runs` 会在结构检查部分结束，这正是迭代期间所需的选项。它不会判断标签是否正确，也不会评价报告质量；这些仍需人工审核。使用 `report_template.tex` 作为最终 `report.tex` 的起点。报告应回答第 1–4 部分的每个问题，并包含足以复现实验的详细信息。

还应包含 `AI_USAGE.md` 文件，详细说明在本作业中使用的任何 AI 技术。列出使用过的所有工具，并清楚描述每个工具的用途。如果没有使用 AI 辅助，也请明确声明。我们不会为提交的 `AI_USAGE.md` 评分，但会通过测验检查你对所提交代码的理解（具体安排已发布在 Piazza）。

最终成绩的一部分取决于最终验证器在私有数据集上的 macro-MCC。私有数据集可能包含种子运行中没有出现的任务和图表类型，因此应避免过拟合公开示例。

| 权重 | 组成部分 |
| ---: | --- |
| 40% | 报告和设计依据 |
| 30% | 验证器在私有集上的 macro-MCC |
| 20% | 测验/理解检查 |
| 10% | 代码质量和可复现性 |
