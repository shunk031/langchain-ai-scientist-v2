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
You are an experienced AI researcher analyzing experimental results. You have been provided with plots from a machine learning experiment. 

Please select 10 most relevant plots to analyze. For similar plots (e.g. generated samples at each epoch), select only at most 5 plots at a suitable interval of epochs. Format your response as a list of plot paths, where each plot path includes the full path to the plot file.
"""

PLOT_PATHS_PROMPT: Final[str] = """\
# Plot Paths

{plot_paths}
"""


class PlotSelectionSpec(BaseModel):
    """Select the 10 most relevant plots for analysis."""

    selected_plots: List[str] = Field(
        description="Full path to a plot file",
        max_length=10,
    )


def select_generated_plots(
    state: TreeBasedExperimentationState, config: RunnableConfig
):
    from . import AgentConfig

    conf = AgentConfig.from_runnable_config(config)

    llm = init_chat_model(
        model=conf.feedback.model_name,
        temperature=conf.feedback.temperature,
    )
    system_prompt = SystemMessage(SYSTEM_PROMPT)
    human_prompt = HumanMessagePromptTemplate.from_template(PLOT_PATHS_PROMPT)

    prompt = ChatPromptTemplate.from_messages(messages=[system_prompt, human_prompt])
    chain = prompt | llm.with_structured_output(PlotSelectionSpec)

    input_dict = {
        "plot_paths": plot_paths,
    }
    output = chain.invoke(input_dict)
    assert isinstance(output, PlotSelectionSpec)

    raise NotImplementedError
