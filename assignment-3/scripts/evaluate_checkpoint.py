"""Evaluate a checkpoint, an exported adapter, or the base model on a
held-out split (given).

Graders run this on the shipped val splits against your exported Part 5
adapter, served with vLLM.

The model settings (base model, chat template, thinking mode, max_tokens,
temperature, stop sequences) come from the same YAML config the run was
trained with, so evaluation renders prompts exactly like training did. The
environment and agent come from the config's `train` section. Any field can
be overridden with `--set`, as in the training scripts.

Where the samples come from, one of:
  --model-path tinker://...      a Tinker checkpoint (honours TINKER_BASE_URL)
  --server URL --model NAME      an OpenAI-compatible /v1/completions server
                                 (vLLM, SGLang) serving NAME
  neither                        the base model, sampled through Tinker

Examples:
  python scripts/evaluate_checkpoint.py --config configs/p2b_number_search_grpo.yaml \\
      --model-path tinker://... --k 4
  python scripts/evaluate_checkpoint.py --config configs/p5_textcraft_grpo.yaml \\
      --tasks data/textcraft_val_hard.jsonl --k 8 --model-path tinker://...
  python scripts/evaluate_checkpoint.py --config configs/p5_textcraft_grpo.yaml \\
      --set model.enable_thinking=false      # base model, thinking off

  # An adapter exported with scripts/export_checkpoint.py, served by vLLM:
  vllm serve Qwen/Qwen3.5-4B --enable-lora --max-lora-rank 32 \\
      --lora-modules a3=<hf_model_id>
  python scripts/evaluate_checkpoint.py --config configs/p5_textcraft_grpo.yaml \\
      --server http://localhost:8000/v1 --model a3 --k 4
  # The same with SGLang (the adapter is named base:adapter in --model):
  python -m sglang.launch_server --model-path Qwen/Qwen3.5-4B --enable-lora \\
      --lora-paths a3=<hf_model_id> --max-lora-rank 32 --port 8000
  python scripts/evaluate_checkpoint.py --config configs/p5_textcraft_grpo.yaml \\
      --server http://localhost:8000/v1 --model Qwen/Qwen3.5-4B:a3 --k 4

The printed JSON carries the results plus `model_path`, `server` and
`model`, so a saved output says what was evaluated.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from scottyrl.agents import AGENT_REGISTRY
from scottyrl.config import load_config
from scottyrl.envs import make_env_factory
from scottyrl.evaluate import evaluate_policy
from scottyrl.policy import CompletionsServerPolicy, TinkerPolicy, load_chat_template
from scottyrl.tinker_adapter import make_service_client
from scottyrl.types import load_tasks

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True, help="YAML config (see configs/)")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="SECTION.KEY=VALUE", help="override a config field")
    p.add_argument("--model-path", default=None,
                   help="tinker checkpoint path; omit (and no --server) for the base model")
    p.add_argument("--server", default=None, metavar="URL",
                   help="OpenAI-compatible API root, e.g. http://localhost:8000/v1 (needs --model)")
    p.add_argument("--model", default=None, metavar="NAME",
                   help="model (adapter) name the server knows, e.g. a3 (needs --server)")
    p.add_argument("--tasks", default=None, help="jsonl path; defaults to the env's val split")
    p.add_argument("--k", type=int, default=None, help="rollouts per task (default: train.eval_k)")
    p.add_argument("--num-tasks", type=int, default=None, help="evaluate only the first N tasks")
    p.add_argument("--concurrency", type=int, default=64)
    args = p.parse_args(argv)
    if args.model_path and args.server:
        p.error("--model-path and --server are alternatives; give one of them")
    if bool(args.server) != bool(args.model):
        p.error("--server and --model go together")
    return args


async def main() -> None:
    args = parse_args()

    config = load_config(args.config, args.overrides)
    model_cfg, train_cfg = config.model, config.train
    k = args.k if args.k is not None else train_cfg.eval_k

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.base_model)
    rendering = dict(
        max_tokens=model_cfg.max_tokens,
        temperature=model_cfg.temperature,
        chat_template=load_chat_template(model_cfg.chat_template),
        enable_thinking=model_cfg.enable_thinking,
        preserve_thinking_history=model_cfg.preserve_thinking_history,
        stop_sequences=model_cfg.stop_sequences,
    )
    if args.server:
        policy = CompletionsServerPolicy(args.server, args.model, tokenizer, **rendering)
    else:
        service_client = make_service_client()  # honours TINKER_BASE_URL
        if args.model_path:
            sampling_client = service_client.create_sampling_client(model_path=args.model_path)
        else:
            sampling_client = service_client.create_sampling_client(base_model=model_cfg.base_model)
        policy = TinkerPolicy(sampling_client, tokenizer, **rendering)

    tasks = load_tasks(args.tasks or DATA_DIR / f"{train_cfg.env}_val.jsonl")
    if args.num_tasks:
        tasks = tasks[: args.num_tasks]

    results = await evaluate_policy(
        policy,
        tasks,
        make_env_factory(train_cfg.env),
        k=k,
        concurrency=args.concurrency,
        agent_cls=AGENT_REGISTRY[train_cfg.agent],
    )
    if args.server:
        results["model_path"] = None
        results["server"] = args.server
        results["model"] = args.model
    else:
        results["model_path"] = args.model_path or model_cfg.base_model
        results["server"] = None
        results["model"] = None
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
