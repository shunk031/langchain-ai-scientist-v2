from typing import Final

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableConfig

from ai_scientist_v2.pipelines.tree_based_experimentation import (
    TreeBasedExperimentationState,
)

SYSTEM_PROMPT: Final[str] = """\
# Introduction

You are an AI researcher analyzing experimental results stored in numpy files. Write code to load and analyze the metrics from `experiment_data.npy`.

# Original Code

{original_code}

# Instructions

0. Make sure to get the working directory from `os.path.join(os.getcwd(), "working")`
1. Load the `experiment_data.npy file`, which is located in the working directory
2. Extract metrics for each dataset. Make sure to refer to the original code to understand the structure of the data
3. Always print the name of the dataset before printing the metrics
4. Always print the name of the metric before printing the value by specifying the metric name clearly. Avoid vague terms like 'train,' 'val,' or 'test.' Instead, use precise labels such as 'train accuracy,' 'validation loss,' or 'test F1 score,' etc.
5. You only need to print the best or final value for each metric for each dataset
6. DO NOT CREATE ANY PLOTS

Important code structure requirements:

- Do NOT put any execution code inside 'if __name__ == \"__main__\":' block. Do not use 'if __name__ == \"__main__\":' at all.
- All code should be at the global scope or in functions that are called from the global scope
- The script should execute immediately when run, without requiring any special entry point
"""


EXTRACTION_PROMPT: Final[str] = """\
# Response Format

Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), followed by a single markdown code block (using the format ```python ... ```) which implements the full code for the metric parsing.

There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. Your generated code should be complete and executable.
"""


def extract_metrics_node(state: TreeBasedExperimentationState, config: RunnableConfig):
    from . import AgentConfig

    conf = AgentConfig.from_runnable_config(config)

    llm = init_chat_model(
        model=conf.code.model_name,
        temperature=conf.code.temperature,
    )

    raise NotImplementedError
