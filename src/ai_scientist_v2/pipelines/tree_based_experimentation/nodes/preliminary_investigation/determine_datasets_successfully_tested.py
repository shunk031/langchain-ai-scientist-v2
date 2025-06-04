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

You are an AI researcher analyzing experiment results. Based on the plot analyses and feedback, determine which datasets are successfully tested. Return reasoning and the dataset names that are successfully executed, or an empty string if no datasets are successfully executed.

# Plot Analysis

{plot_analysis}

# VLM Feedback Summary

{vlm_feedback_summary}

# Original plotting code

```python
{plotting_code}
```

# Response Format

Your response should start with 'REASONING: <reasoning>' to think about the plot analysis and feedback in the first line. In the second line, you should have a list of dataset names that are successfully executed, starting with 'SUCCESSFULLY_TESTED_DATASETS: 

{list_datasets_successfully_tested}
"""

FEEDBACK_PROMPT: Final[str] = """\
Provide a detailed evaluation of completion status.
"""


def determine_datasets_successfully_tested_node(
    state: TreeBasedExperimentationState, config: RunnableConfig
):
    conf = AgentConfig.from_runnable_config(config)

    llm = init_chat_model(
        model=conf.feedback.model_name,
        temperature=conf.feedback.temperature,
    )
    system_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT)
    human_prompt = HumanMessage(FEEDBACK_PROMPT)
    prompt = ChatPromptTemplate.from_messages(
        messages=[system_prompt, human_prompt],
    )
    chain = prompt | llm

    input_dict = {
        "vlm_feedback": state.vlm_feedback,
        "datasets_successfully_tested": state.datasets_successfully_tested,
    }
    output = chain.invoke(input_dict)

    raise NotImplementedError
