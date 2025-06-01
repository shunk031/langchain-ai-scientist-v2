import json
import pathlib
from typing import Dict, List, Optional, Union

from langchain_core.messages import AnyMessage, ToolMessage
from loguru import logger


def load_ideas(ideas_file: pathlib.Path):
    """
    Load ideas from a JSONL file.
    """
    logger.debug(f"Loading ideas from {ideas_file}")
    with open(ideas_file, "r") as f:
        ideas = [json.loads(line) for line in f]
    return ideas


def load_workshop_description(workshop_file: pathlib.Path) -> str:
    """
    Load the workshop description from a markdown file.
    """
    logger.debug(f"Loading workshop description from {workshop_file}")
    with open(workshop_file, "r") as f:
        workshop_description = f.read()
    return workshop_description


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
