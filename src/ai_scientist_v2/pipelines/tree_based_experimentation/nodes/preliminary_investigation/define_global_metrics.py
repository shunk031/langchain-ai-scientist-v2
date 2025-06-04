from typing import Final

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.runnables import RunnableConfig

from ai_scientist_v2.pipelines.tree_based_experimentation import (
    EvaluationMetrics,
    TreeBasedExperimentationState,
)

SYSTEM_PROMPT: Final[str] = """\
# Introduction
You are an AI researcher setting up experiments.
Please propose meaningful evaluation metrics that will help analyze the performance and characteristics of solutions for this research task.
"""

DEFINE_GLOBAL_METRICS_PROMPT: Final[str] = """\
# Research Idea

You are an ambitious AI researcher who is looking to publish a paper that will contribute significantly to the field. You have an idea and you want to conduct creative experiments to gain scientific insights. Your aim is to run experiments to gather sufficient results for a top conference paper. Your research idea is as follows:

## Title

{title}

## Abstract

{abstract}

## Short Hypothesis

{short_hypothesis}

## Code to Use

```python
{code_to_use}
```

# Current Main Stage: {main_stage_name}
## Sub-stage: {sub_stage_num} - {sub_stage_name}
### Sub-stage goals
{sub_stage_goals}

# Instructions

Propose a single evaluation metric that would be useful for analyzing the performance of solutions for this research task.
Note: Validation loss will be tracked separately so you don't need to include it in your response.

Format your response as a following json object:

{{
    "name": "The name of the metric",
    "maximize": "Whether higher values are better (true/false)",
    "description": "A brief explanation of what the metric measures"
}}

Your json should contain only one metric.
"""


def define_global_metrics_node(
    state: TreeBasedExperimentationState, config: RunnableConfig
) -> TreeBasedExperimentationState:
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
    chain = prompt | llm.with_structured_output(EvaluationMetrics)

    input_dict = {
        "title": conf.idea.title,
        "abstract": conf.idea.abstract,
        "short_hypothesis": conf.idea.short_hypothesis,
        "main_stage_name": state.main_stage_name,
        "sub_stage_num": state.sub_stage_num,
        "sub_stage_name": state.sub_stage_name,
        "sub_stage_goals": state.sub_stage_goals,
        "code_to_use": state.code_to_use,
    }
    output = chain.invoke(input_dict)
    assert isinstance(output, EvaluationMetrics)

    return state.model_copy(update={"evaluation_metrics": output})
