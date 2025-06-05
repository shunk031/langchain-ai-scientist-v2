from typing import Dict, List, Union

import pytest
from langchain_core.runnables import RunnableConfig

from ai_scientist_v2.pipelines.tree_based_experimentation import (
    EvaluationMetrics,
    TreeBasedExperimentationState,
)
from ai_scientist_v2.pipelines.tree_based_experimentation.nodes import (
    define_global_metrics_node,
)
from ai_scientist_v2.pipelines.tree_based_experimentation.pipeline import STAGE1_GOAL


@pytest.fixture
def code_to_use() -> str:
    return "No code is provided for this stage."


def test_define_global_metrics_node(
    idea: List[Dict[str, Union[str, List[str]]]], code_to_use: str
):
    state = TreeBasedExperimentationState(
        messages=[],
        main_stage_name="Initial Implementation",
        sub_stage_name="Preliminary Investigation",
        sub_stage_num=1,
        sub_stage_goals=STAGE1_GOAL,
        code_to_use=code_to_use,
    )
    config = RunnableConfig({"configurable": {"idea": idea}})

    output = define_global_metrics_node(state=state, config=config)

    assert isinstance(output, TreeBasedExperimentationState)
    assert isinstance(output.evaluation_metrics, EvaluationMetrics)
