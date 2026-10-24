# 数据集来源（请勿重新生成或编辑）

评分器会使用以下确切的数据划分进行评估。

| 文件 | 内容 |
|---|---|
| `number_search_train.jsonl` | 50,000 个 Number Search 任务（训练集） |
| `number_search_val.jsonl` | 1,000 个留出任务 |
| `textcraft_train.jsonl` | 852 个 TextCraft-Synth 任务，**中等**难度（训练集） |
| `textcraft_val.jsonl` | 213 个留出的中等难度任务 |
| `textcraft_val_hard.jsonl` | 136 个留出的**高难度**任务（用于测试泛化能力） |
| `textcraft_recipes.json` | 完整的合成配方书（1,452 个物品）；所有 TextCraft 任务都基于它生成 |

注意：

- Number Search 的 `max_steps` 为 `ceil(log2(high - low + 2))`，这是包含最后一次正确猜测在内的精确最坏情况二分查找预算。没有额外猜测；每个 LLM 轮次使用一步。
- TextCraft 任务在 `task.info` 中提供 `target_items` 和 `initial_inventory`；原材料只能来自初始物品栏（不存在 gather 操作），并且必须调用 `finish()` 才能获得奖励。
