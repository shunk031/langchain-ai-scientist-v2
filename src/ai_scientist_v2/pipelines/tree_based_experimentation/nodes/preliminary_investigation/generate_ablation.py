from typing import Final

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from ai_scientist_v2.pipelines.tree_based_experimentation import (
    EvaluationMetrics,
    TreeBasedExperimentationState,
)

SYSTEM_PROMPT: Final[str] = """\
# Introduction

You are an AI researcher conducting ablation studies. Based on the current implementation and previous ablations (if any), propose ONE new ablation study that tests a different aspect of the model.

# Base Code You are Working On

```python
{base_code}
```

# Previous Ablations

The following ablations have already been tried and should not be repeated:

{completed}

# Instructions

Requirements for the new ablation:

1. Identify ONE specific component/feature to ablate
2. Ensure the ablation is different from previous completed or running attempts
3. The ablation should be a new idea, not a variation of previous ideas
4. If you have only used a single synthetic dataset throughout the experiment, one of your ablations should be to use multiple synthetic datasets (at least 3 different datasets)
"""

ABLATION_PROMPT: Final[str] = """\
# Response Format

Format your response as a JSON object with the following fields:

{{
    "ablation_name": "represent the name of the ablation",
    "ablation_description": "represent a brief description of what component is being ablated and why (3-5 sentences)",
}}
"""


class AblationIdeaSpec(BaseModel):
    """Propose a new ablation study idea."""

    ablation_name: str = Field(
        description="Name of the ablation",
    )
    ablation_description: str = Field(
        description="Brief description of what component is being ablated and why",
    )


def generate_ablation_node(state, config):
    from . import AgentConfig

    conf = AgentConfig.from_runnable_config(config)

    llm = init_chat_model(
        model=conf.code.model_name,
        temperature=conf.code.temperature,
    )

    system_prompt = SystemMessage(
        SYSTEM_PROMPT,
    )
    human_prompt = HumanMessagePromptTemplate.from_template(
        DEFINE_GLOBAL_METRICS_PROMPT
    )
    prompt = ChatPromptTemplate.from_messages(messages=[system_prompt, human_prompt])
    chain = prompt | llm.with_structured_output(AblationIdeaSpec)

    raise NotImplementedError
