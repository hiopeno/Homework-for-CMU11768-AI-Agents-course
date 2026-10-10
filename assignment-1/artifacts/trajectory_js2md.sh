#!/usr/bin/env bash
# Write a readable Markdown report for an agent trajectory.
# Usage:
#   ./show_trajectory.sh [trajectory.json] [--full]

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
trajectory_path="$script_dir/part1-trajectory.json"
show_full=false

for argument in "$@"; do
    case "$argument" in
        --full)
            show_full=true
            ;;
        --help|-h)
            sed -n '2,5p' "$0"
            exit 0
            ;;
        -*)
            printf 'Unknown option: %s\n' "$argument" >&2
            exit 2
            ;;
        *)
            trajectory_path="$argument"
            ;;
    esac
done

if [[ ! -f "$trajectory_path" ]]; then
    printf 'Trajectory file not found: %s\n' "$trajectory_path" >&2
    exit 1
fi

timestamp=$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M')
trajectory_filename=$(basename -- "$trajectory_path")
trajectory_stem=${trajectory_filename%.*}
output_path="$script_dir/trajectory_md/${trajectory_stem}_${timestamp}.md"
temporary_path=$(mktemp "$script_dir/.show_trajectory.XXXXXX")
trap 'rm -f "$temporary_path"' EXIT

python3 - "$trajectory_path" "$show_full" <<'PY' > "$temporary_path"
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
show_full = sys.argv[2].lower() == "true"
MAX_CHARS = 3_000


def clipped(text: object) -> str:
    """Keep normal results intact while making large terminal dumps readable."""
    text = str(text or "")
    if show_full or len(text) <= MAX_CHARS:
        return text
    omitted = len(text) - MAX_CHARS
    return text[:MAX_CHARS] + f"\n... [{omitted} characters omitted; rerun with --full]"


def code_block(text: object, language: str = "text") -> str:
    """Use a fence longer than any backtick run already present in the text."""
    text = str(text or "")
    longest_backtick_run = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest_backtick_run + 1)
    return f"{fence}{language}\n{text}\n{fence}"


def tag_value(content: str, name: str) -> str | None:
    opening = f"<{name}>"
    closing = f"</{name}>"
    start = content.find(opening)
    end = content.find(closing, start + len(opening))
    if start < 0 or end < 0:
        return None
    return content[start + len(opening):end]


def render_observation(content: object) -> str:
    """Remove duplicated executor fields while retaining failures and output."""
    text = str(content or "")
    output = tag_value(text, "output")
    stdout = tag_value(text, "stdout")
    stderr = tag_value(text, "stderr")
    exception = tag_value(text, "exception_info")
    returncode = tag_value(text, "returncode")

    # Environment.execute records both output and stdout; output is sufficient.
    if output is not None or stdout is not None or stderr is not None:
        parts: list[str] = []
        if returncode not in (None, "", "0"):
            parts.append(f"exit code: {returncode}")
        if exception:
            parts.append(f"exception: {exception}")
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        if output is not None:
            parts.append(f"output:\n{output}")
        elif stdout:
            parts.append(f"stdout:\n{stdout}")
        return clipped("\n".join(parts) or "(no output)")
    return clipped(text) or "(no output)"


def without_none_values(value: object) -> object:
    """The SDK serializes omitted response fields as null; prompts omit them."""
    if isinstance(value, dict):
        return {
            key: without_none_values(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [without_none_values(item) for item in value]
    return value


def tool_observations(next_prompt: list[dict], response_message: dict) -> list[dict]:
    """Find the contiguous tool observations added after this response.

    Call IDs are intentionally not used: providers may reuse an ID in a later
    turn, whereas ordering in the next prompt is unambiguous.
    """
    assistant_positions = [
        index
        for index, message in enumerate(next_prompt)
        if (
            message.get("role") == "assistant"
            and without_none_values(message) == without_none_values(response_message)
        )
    ]
    if not assistant_positions:
        return []
    observations: list[dict] = []
    for message in next_prompt[assistant_positions[-1] + 1:]:
        if message.get("role") != "tool":
            break
        observations.append(message)
    return observations


def render_tool_call(number: int, call: dict) -> None:
    function = call.get("function") or {}
    name = function.get("name", "unknown")
    raw_arguments = function.get("arguments", "")
    print(f"{number}. **`{name}`**")
    try:
        arguments = json.loads(raw_arguments)
    except (TypeError, json.JSONDecodeError):
        print(code_block(f"Invalid JSON arguments:\n{raw_arguments}"))
        return
    if name == "execute" and isinstance(arguments, dict) and isinstance(arguments.get("command"), str):
        print(code_block(arguments["command"], "bash"))
    else:
        print(code_block(json.dumps(arguments, ensure_ascii=False, indent=2), "json"))


try:
    trajectory = json.loads(path.read_text(encoding="utf-8"))
except json.JSONDecodeError as exc:
    raise SystemExit(f"Invalid JSON in {path}: {exc}")

prompts = trajectory.get("prompts")
responses = trajectory.get("responses")
if not isinstance(prompts, list) or not isinstance(responses, list):
    raise SystemExit("Trajectory must contain list fields named 'prompts' and 'responses'.")

print("# Agent Trajectory")
print()
print(f"- Source: `{path.name}`")
print(f"- Model requests: **{len(responses)}**")
if len(prompts) != len(responses):
    print(f"- Warning: {len(prompts)} prompts but {len(responses)} responses")

if prompts and isinstance(prompts[0], list):
    system_prompt = next(
        (item.get("content", "") for item in prompts[0] if item.get("role") == "system"),
        "",
    )
    task = next(
        (item.get("content", "") for item in prompts[0] if item.get("role") == "user"),
        "",
    )
    if system_prompt:
        print("\n## System Prompt\n")
        print(code_block(system_prompt, "text"))
    print("\n## Task\n")
    print(code_block(clipped(task), "markdown"))

for index, response in enumerate(responses):
    choices = response.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    usage = response.get("usage") or {}
    print(f"\n## Round {index + 1}\n")
    if isinstance(usage.get("prompt_tokens"), int):
        completion = usage.get("completion_tokens", "?")
        print(f"**Tokens:** prompt {usage['prompt_tokens']}, completion {completion}\n")

    reasoning = message.get("reasoning_content")
    if reasoning:
        print("### Reasoning\n")
        print(code_block(clipped(reasoning)))
    content = message.get("content")
    if content:
        print("\n### Assistant Message\n")
        print(code_block(clipped(content)))

    calls = message.get("tool_calls") or []
    if calls:
        print("\n### Tool Calls\n")
        for number, call in enumerate(calls, start=1):
            render_tool_call(number, call)
            print()
    elif not content:
        print("_Empty response; no tool call._")

    if index + 1 < len(prompts) and isinstance(prompts[index + 1], list):
        observations = tool_observations(prompts[index + 1], message)
        if observations:
            print("### Observations\n")
            for number, observation in enumerate(observations, start=1):
                print(f"#### Observation {number}\n")
                print(code_block(render_observation(observation.get("content"))))
                print()

compactions = trajectory.get("compactions") or []
if compactions:
    print("## Context Compactions\n")
    for event in compactions:
        print(
            "- Step {step}: estimated tokens {before} -> {after}".format(
                step=event.get("step", "?"),
                before=event.get("estimated_tokens_before", "?"),
                after=event.get("estimated_tokens_after", "?"),
            )
        )
PY

mv -- "$temporary_path" "$output_path"
trap - EXIT
printf 'Markdown trajectory written to: %s\n' "$output_path"
