from typing import Final

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableConfig

from ai_scientist_v2.pipelines.tree_based_experimentation import (
    EvaluationMetrics,
    TreeBasedExperimentationState,
)

from . import AgentConfig

SYSTEM_PROMPT: Final[str] = """\
# Introduction

You are an AI researcher analyzing experimental results.

# Research Idea

{research_idea}

# Implementation

{code}

# Plan

{plan}

# Execution Output

{execution_output}

# Analysis

{analysis}

# Metrics

{metrics}

# Plot Analysis

{plot_analysis}

# VLM Feedback

{vlm_feedback_summary}
"""

SUMMARIZE_PROMPT: Final[str] = """\
Please summarize the findings from this experiment iteration.
"""


def generate_node_summary_node(
    state: TreeBasedExperimentationState, config: RunnableConfig
):
    conf = AgentConfig.from_runnable_config(config)

    llm = init_chat_model(
        model=conf.feedback.model_name,
        temperature=conf.feedback.temperature,
    )
    system_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT)
    human_prompt = HumanMessage(SUMMARIZE_PROMPT)
    prompt = ChatPromptTemplate.from_messages(
        messages=[system_prompt, human_prompt],
    )
    chain = prompt | llm

    input_dict = {
        "research_idea": conf.idea.title,
        "code": state.code_to_use,
        "plan": state.sub_stage_goals,
        "execution_output": state.execution_output,
        "analysis": state.analysis,
        "metrics": state.metrics,
        "plot_analysis": state.plot_analysis,
        "vlm_feedback_summary": state.vlm_feedback_summary,
    }
    output = chain.invoke(input_dict)

    raise NotImplementedError
