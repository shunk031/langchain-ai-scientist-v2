from typing import Final, List

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from ai_scientist_v2.pipelines.tree_based_experimentation import (
    TreeBasedExperimentationState,
)

SYSTEM_PROMPT: Final[str] = """\
# Introduction

You are an experienced AI researcher. Your previous code for research experiment had a bug, so based on the information below, you should revise it in order to fix this bug. Your response should be an implementation outline in natural language, followed by a single markdown code block which implements the bugfix/solution.

# Research Idea

{research_idea}

# Previous (buggy) implementation

```python
{previous_code}
```

# Execution Output

{execution_output}

# Feedback based on Generated Plots

{vlm_feedback_summary}

# Feedback about execution time

{execution_time_feedback}

# Instructions

## Response Format

Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), followed by a single markdown code block (using the format ```python ... ```) which implements the full code including the bugfix/solution. There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. 

Your generated code should be complete and executable. Do not omit any part of the code, even if it was part of a previous implementation. Make sure to write concise code.

## Bugfix Improvement Sketch Guideline

You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed. Don't suggest to do EDA.

## Implementation Guideline

CRITICAL GPU REQUIREMENTS - Your code MUST include ALL of these:

- At the start of your code, add these lines to handle GPU/CPU:

    ```python
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    ```

- ALWAYS move models to device using the `.to(device)` method
- ALWAYS move input tensors to device using the `.to(device)` method
- ALWAYS move model related tensors to device using the `.to(device)` method
- For optimizers, create them AFTER moving model to device
- When using DataLoader, move batch tensors to device in training loop:

    ```python
    batch = {{k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}}
    ```

CRITICAL MODEL INPUT GUIDELINES:
- Always pay extra attention to the input to the model being properly normalized
- This is extremely important because the input to the model's forward pass directly affects the output, and the loss function is computed based on the output

## Data Overview

{data_overview}
"""


def debug_code_node(state: TreeBasedExperimentationState, config: RunnableConfig):
    raise NotImplementedError
