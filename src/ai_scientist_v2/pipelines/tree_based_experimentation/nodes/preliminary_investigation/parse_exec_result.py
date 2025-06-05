from typing import Final, List

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from ai_scientist_v2.pipelines.tree_based_experimentation import (
    TreeBasedExperimentationState,
)

SYSTEM_PROMPT: Final[str] = """\
# Introduction

Parse the metrics from the execution output. You only need the final or best value of a metric for each dataset, not the entire list during training.
"""

EXTRACTION_PROMPT: Final[str] = """\
# Execution Output

{execution_output}
"""


class DatasetResult(BaseModel):
    dataset_name: str = Field(
        description="The name of the dataset. Never include 'train', 'val', or 'test' in the dataset name."
    )
    final_value: float = Field(
        description="The final value of the metric for this dataset."
    )
    best_value: float = Field(
        description="The best value of the metric for this dataset"
    )


class MetricName(BaseModel):
    metric_name: str = Field(
        description="Specify the metric name clearly. Avoid vague terms like 'train,' 'val,' or 'test.' Instead, use precise labels such as 'train accuracy,' 'validation loss,' or 'test F1 score,' etc."
    )
    lower_is_better: bool = Field(
        description="Whether lower values are better for this metric",
    )
    description: str = Field(description="Description of the metric")
    data: List[DatasetResult]


class MetricParseSpec(BaseModel):
    """Parse the metrics from the execution output."""

    valid_metrics_recieved: bool = Field(
        description="True if the metrics were successfully received, False otherwise. For example if the execution output does not contain any metrics, set this to False."
    )
    metric_names: List[MetricName]


def parse_exec_result_node(
    state: TreeBasedExperimentationState, config: RunnableConfig
):
    from . import AgentConfig

    conf = AgentConfig.from_runnable_config(config)
    llm = init_chat_model(
        model=conf.feedback.model_name,
        temperature=conf.feedback.temperature,
    )
    chain = prompt | llm.with_structured_output(MetricParseSpec)
    raise NotImplementedError
