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

SYSTEM_PROMPT: Final[str] = """\
Evaluate if stage 2 (baseline tuning) is complete based on the following evidence:

## Figure Analysis

{vlm_feedback}

## Datasets Tested

{datasets_successfully_tested}

## Requirements for completion

1. Training curves should show stable convergence.
2. Results should be tested on at least two datasets.
3. No major instabilities or issues in the plots.
"""

FEEDBACK_PROMPT: Final[str] = """\
Provide a detailed evaluation of completion status.
"""


def perform_vlm_feedback_node(
    state: TreeBasedExperimentationState, config: RunnableConfig
):
    from . import AgentConfig

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
