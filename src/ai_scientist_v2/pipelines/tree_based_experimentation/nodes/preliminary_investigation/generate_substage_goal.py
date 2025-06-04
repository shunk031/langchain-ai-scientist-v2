from typing import Annotated, Dict, Final, List, Literal, Optional, Union

from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from ai_scientist_v2.config import BFTSAgentConfig
from ai_scientist_v2.models import Idea
from ai_scientist_v2.pipelines.tree_based_experimentation import (
    EvaluationMetrics,
    TreeBasedExperimentationState,
)
from ai_scientist_v2.utils.configuration import Configuration

SYSTEM_PROMPT: Final[str] = """\
Based on the current experimental progress, generate focused goals for the next sub-stage.

# Main Stage Goals
{main_stage_goal}

# Current Progress

## Total attempts

{total_nodes}

## Successful implementations

{good_nodes}

## Best performance: 

{best_metric_value}

## Convergence status: 

{convergence_status}

# Current Issues

{issues}

# Recent Changes:

{recent_changes}
"""

GENERATE_SUBSTAGE_GOAL_PROMPT: Final[str] = """\
Generate specific, actionable sub-stage goals that:

1. Address current issues and limitations
2. Build on recent progress
3. Move towards main stage goals
4. Are concrete and measurable

Format your response as a JSON object:
{{
    "goals": "Detailed, specific goals for the next sub-stage.",
    "sub_stage_name": "The name of the next sub-stage"
}}
"""


class SubStageGoal(BaseModel):
    goals: str = Field(
        description="Detailed, specific goals for the next sub-stage.",
    )
    sub_stage_name: str = Field(
        description="The name of the next sub-stage",
    )


def generate_substage_goal_node(
    stage: TreeBasedExperimentationState, config: RunnableConfig
):
    conf = AgentConfig.from_runnable_config(config)

    breakpoint()

    raise NotImplementedError
