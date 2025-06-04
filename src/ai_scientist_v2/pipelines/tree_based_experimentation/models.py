from typing import Annotated, List, Literal, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field

MainStageName = Literal[
    "Initial Implementation",
    "Baseline Tuning",
    "Creative Research",
    "Ablation Studies",
]


class EvaluationMetrics(BaseModel):
    name: str = Field(
        description="The name of the metric",
    )
    maximize: bool = Field(
        description="Whether higher values are better (true/false)",
    )
    description: str = Field(
        description="A brief explanation of what the metric measures",
    )


class TreeBasedExperimentationState(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages]

    main_stage_name: MainStageName = "Initial Implementation"

    sub_stage_name: str = "Preliminary Investigation"
    sub_stage_num: int = 1
    sub_stage_goals: str
    code_to_use: str

    evaluation_metrics: Optional[EvaluationMetrics] = None

    vlm_feedback: Optional[str] = None

    datasets_successfully_tested: Optional[List[str]] = None


class Stage(BaseModel):
    name: str
    description: str
    goals: List[str]
    max_iterations: int
    num_drafts: int
    stage_number: int
