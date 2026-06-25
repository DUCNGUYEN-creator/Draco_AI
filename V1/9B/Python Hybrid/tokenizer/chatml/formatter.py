# DracoAI V1 — tokenizer/chatml/formatter.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
DracoAI Tokenizer — ChatML Formatter
======================================
Utilities for formatting message content before encoding.
"""

from typing import Dict, List, Optional


def format_system_prompt(prompt: str) -> str:
    """
    Strip and normalise a system prompt string.

    Parameters
    ----------
    prompt : str
        Raw system prompt text.

    Returns
    -------
    str
        Cleaned system prompt (stripped whitespace, no leading newlines).
    """
    return prompt.strip()


def format_tool_call(name: str, arguments: str) -> str:
    """
    Format a tool call for inclusion in an assistant message.

    Parameters
    ----------
    name : str
        Tool/function name.
    arguments : str
        JSON-serialised arguments string.

    Returns
    -------
    str
        Formatted string: "<tool_call>\\n{"name": ..., "arguments": ...}\\n</tool_call>"
    """
    import json
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"<tool_call>\n{payload}\n</tool_call>"


def format_tool_result(name: str, result: str) -> str:
    """
    Format a tool result for inclusion in a tool role message.

    Parameters
    ----------
    name : str
        Tool/function name.
    result : str
        The tool's return value (stringified).

    Returns
    -------
    str
        Formatted result string.
    """
    return f"<tool_response>\n{result}\n</tool_response>"


def build_messages(
    user_text: str,
    system_prompt: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """
    Build a messages list from convenience arguments.

    Parameters
    ----------
    user_text : str
        The current user message.
    system_prompt : Optional[str]
        Optional system-level instruction.
    history : Optional[List[Dict[str, str]]]
        Prior conversation turns (list of {"role", "content"} dicts).

    Returns
    -------
    List[Dict[str, str]]
        Complete messages list ready for ChatMLTemplate.encode().
    """
    messages: List[Dict[str, str]] = []

    if system_prompt:
        messages.append({"role": "system", "content": format_system_prompt(system_prompt)})

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": user_text})
    return messages