import json
from typing import Dict, List, Optional, Union

from langchain_core.messages import AnyMessage, ToolMessage


def jsonl_to_prompt_string(jsonl: List[Dict[str, Union[str, List[str]]]]) -> str:
    """Converts a list of JSON objects to a formatted string for prompt usage."""
    json_strings = [json.dumps(item) for item in jsonl]
    return "\n\n".join(json_strings)


def get_last_tool_message(messages: List[AnyMessage]) -> Optional[ToolMessage]:
    """Extracts the last tool message from a list of messages."""
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            return message

    return None


def get_last_tool_result(messages: List[AnyMessage]) -> str:
    """Extracts the content of the last tool message from a list of messages."""
    last_tool_message = get_last_tool_message(messages)
    last_tool_result = (
        last_tool_message.content if last_tool_message else "No tool results found."
    )
    assert isinstance(last_tool_result, str)
    return last_tool_result
