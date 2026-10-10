"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations

import json
from typing import Any

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
    format_tool_output,
)
from assignment.agent.tools import EXECUTE_TOOL, INVOKE_SKILL_TOOL, SEND_MESSAGE_TOOL
from assignment.env import Environment

tool_list = [EXECUTE_TOOL, SEND_MESSAGE_TOOL]#INVOKE_SKILL_TOOL由父类按需添加，所以这里不加入
class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch."""

    def __init__(
        self,
        task: str,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ):
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )
        self.task = task
        self.submitted_patch = ""

        # TODO(Part 1.3): Make the `execute` and `send_message` tools available
        # to the agent.

        # TODO(1.1.b): Construct the system prompt and task_prompt. These
        # should be usable by the `Agent.build_prompt` method.
        system_information = json.dumps(
            {
                "machine": environment.machine,
                "release": environment.release,
                "system": environment.system,
                "version": environment.version,
            },
            indent=2,
        )
        self.system_prompt = (
            "You are a coding agent working in a terminal sandbox. Investigate "
            "the reported issue, make a focused fix, and verify it with relevant "
            "commands. Use the available tools to inspect the environment and "
            "observe command results before making decisions.\n\n"
            "<system_information>\n"
            f"{system_information}\n"
            "</system_information>"
        )
        self.task_prompt = task

        self.tools.extend(tool_list)
        if self.skills:
            catalog = "\n".join(
                self.skills[name]["metadata"] for name in sorted(self.skills)
            )
            self.system_prompt += (
                "\n\nReusable skills are available. Before starting work that a skill "
                "covers, call `invoke_skill` with its name and follow the returned "
                "instructions.\n\n<skills>\n"
                f"{catalog}\n</skills>"
            )

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute ``execute`` and ``send_message`` calls in the code sandbox."""

        # TODO(Part 1.3): Parse each call, execute recognized tools, and return
        # one message per call (there may be multiple tool calls in one agent
        # response!). Malformed JSON and unknown tools must become recoverable
        # observations relayed to the agent instead of exceptions.

        output_list: list[dict[str, str]] = []
        for call in tool_calls:
            # 以下是我初次错误实现，混淆了工具定义/schema和openai返回的标准数据格式
            # call_name=call.function.name
            # if call_name == EXECUTE_TOOL.function.name :
            #     bash(call_name,call.function.command)
            # if call_name == SEND_MESSAGE_TOOL.function.name :
            #     bash(echo,call_argu)
            # ################################################################

            
            # 每个工具调用都单独处理；一个调用失败不能阻断同一响应中的其他调用。
            call_id = call.get("id", "") if isinstance(call, dict) else ""
            try:
                function = call.get("function", {}) if isinstance(call, dict) else {}
                if not isinstance(function, dict):
                    raise ValueError("tool function must be an object")

                name = function.get("name")
                arguments_text = function.get("arguments", "")
                arguments = json.loads(arguments_text)
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be a JSON object")

                if name == EXECUTE_TOOL["function"]["name"]:
                    result = self.env.execute(**arguments)
                    content = format_tool_output(result)
                elif name == SEND_MESSAGE_TOOL["function"]["name"]:
                    # send_message 直接把摘要作为观察结果，不需要执行 shell 命令。
                    summary = arguments.get("summary")
                    if not isinstance(summary, str):
                        raise ValueError("summary must be a string")
                    self.finished = True
                    content = summary
                elif name == INVOKE_SKILL_TOOL["function"]["name"]:
                    skill_name = arguments.get("name")
                    if not isinstance(skill_name, str):
                        raise ValueError("name must be a string")
                    skill = self.skills.get(skill_name)
                    if skill is None:
                        raise ValueError(f"unknown skill: {skill_name}")
                    content = skill["content"]
                else:
                    content = f"<tool_error>Unknown tool: {name}</tool_error>"
            except Exception as exc:
                # 把错误作为观察结果交给模型，让它有机会修正下一步调用。
                content = f"<tool_error>{type(exc).__name__}: {exc}</tool_error>"

            output_list.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content,
                }
            )
        return output_list
    
        # #以上是codex帮我补充错误审查机制后的代码，以下是我原始的代码：
        # output_list:list[dict[str, str]]=[]
        # for call in tool_calls:
        #     # 以下是我初次错误实现，混淆了工具定义/schema和openai返回的标准数据格式
        #     # call_name=call.function.name
        #     #     bash(echo,call_argu)
        #     # ################################################################

        #     name = call["function"]["name"]
        #     arguments_text = call["function"]["arguments"]
        #     arguments = json.loads(arguments_text)
         
        #     if name == EXECUTE_TOOL["function"]["name"]:
        #         result = self.env.execute(**arguments)
        #         content = format_tool_output(result)
        #     elif name == SEND_MESSAGE_TOOL["function"]["name"]:
        #         #result = self.env.execute(f"echo {arguments}")#这是我的初次错误实现

        #         summary = arguments["summary"]
        #         self.finished = True
        #         content = summary

        #     output_list.append(
        #         {
        #             "role": "tool",
        #             "tool_call_id": call["id"],
        #             "content": content,
        #         }
        #     )
        # return output_list
        # ############################################################################
