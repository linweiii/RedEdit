"""
fncall.py — Lightweight function-call → text formatting.

RedEdit only needs `convert_fncall_to_text` to pretty-print assistant /
function-call messages when driving the optional qwen-agent ``Assistant``
loop. That helper lives in ``qwen_agent.gui.utils``, which requires the
heavy ``qwen-agent[gui]`` extra (gradio, modelscope_studio, ...) at import
time — even though none of the GUI code is ever used by the attack.

This module re-implements the single function we use, keeping the identical
behaviour and output format. The constants are string literals with the same
values as ``qwen_agent.llm.schema`` so both can be mixed freely.

Upstream reference (Apache-2.0, Copyright 2023 The Qwen team, Alibaba Group):
  qwen_agent/gui/utils.py  →  convert_fncall_to_text()
"""

from __future__ import annotations

from typing import Dict, List

# Same values as qwen_agent.llm.schema (verified against qwen-agent 0.0.34).
ASSISTANT = "assistant"
CONTENT = "content"
FUNCTION = "function"
NAME = "name"
ROLE = "role"
SYSTEM = "system"
USER = "user"
REASONING_CONTENT = "reasoning_content"

_THINK = """
<details>
  <summary>Thinking ...</summary>
{thought}
</details>
"""

_TOOL_CALL = """
<details>
  <summary>Start calling tool "{tool_name}" ...</summary>
{tool_input}
</details>
"""

_TOOL_OUTPUT = """
<details>
  <summary>Finished tool calling.</summary>
{tool_output}
</details>

"""


def convert_fncall_to_text(messages: List[Dict]) -> List[Dict]:
    """Faithful port of ``qwen_agent.gui.utils.convert_fncall_to_text``.

    Converts a list of raw chat-completion responses (which may contain
    structured tool-call content) into human-readable text messages, exactly
    matching the upstream output format.
    """
    new_messages = []

    for msg in messages:
        role, content, reasoning_content, name = (
            msg[ROLE], msg[CONTENT], msg.get(REASONING_CONTENT, ""), msg.get(NAME, None)
        )

        # Handle content as list or string
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if "text" in item:
                        text_parts.append(item["text"])
                    elif "image" in item:
                        data_url = item.get("image", "")
                        text_parts.append(f'<img src="{data_url}" style="max-width:100%;height:auto;" />')
                    elif "audio" in item:
                        text_parts.append(f"[Audio: {item.get('audio', '')}]")
                elif isinstance(item, str):
                    text_parts.append(item)
            content = " ".join(text_parts)
        else:
            content = content or ""

        content = content.lstrip("\n").rstrip().replace("```", "")

        if role in (SYSTEM, USER):
            new_messages.append({ROLE: role, CONTENT: content, NAME: name})

        elif role == ASSISTANT:
            if reasoning_content:
                thought = reasoning_content
                content = _THINK.format(thought=thought) + content

            if "<think>" in content:
                ti = content.find("<think>")
                te = content.find("</think>")
                if te == -1:
                    te = len(content)
                thought = content[ti + len("<think>"):te]
                if thought.strip():
                    _content = content[:ti] + _THINK.format(thought=thought)
                else:
                    _content = content[:ti]
                if te < len(content):
                    _content += content[te:]
                content = _content.strip("\n")

            fn_call = msg.get(f"{FUNCTION}_call", {})
            if fn_call:
                f_name = fn_call["name"]
                f_args = fn_call["arguments"]
                content += _TOOL_CALL.format(tool_name=f_name, tool_input=f_args)
            if len(new_messages) > 0 and new_messages[-1][ROLE] == ASSISTANT and new_messages[-1][NAME] == name:
                new_messages[-1][CONTENT] += content
            else:
                new_messages.append({ROLE: role, CONTENT: content, NAME: name})

        elif role == FUNCTION:
            assert new_messages[-1][ROLE] == ASSISTANT
            new_messages[-1][CONTENT] += _TOOL_OUTPUT.format(tool_output=content)

        else:
            raise TypeError(f"Unknown message role: {role!r}")

    return new_messages
