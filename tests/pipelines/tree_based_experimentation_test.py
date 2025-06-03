import pathlib
from typing import Dict, Final, List

import pytest
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

from ai_scientist_v2.pipelines import (
    create_tree_based_experimentation_pipeline,
    run_tree_based_experimentation_pipeline,
)


@pytest.fixture
def idea(ideas: List[Dict[str, str]]) -> Dict[str, str]:
    return ideas[0]


@pytest.fixture
def global_metrics_definer_model_name() -> str:
    return "openai:gpt-4o"


@pytest.fixture
def temperature() -> float:
    return 1.0


@pytest.fixture
def main_stage_name() -> str:
    return "Initial Implementation"


@pytest.fixture
def sub_stage_num() -> int:
    return 1


@pytest.fixture
def sub_stage_name() -> str:
    return "Preliminary Investigation"


@pytest.fixture
def code_to_use() -> str:
    return """No code is provided for this stage."""


def test_create_tree_based_experimentation_pipeline(root_dir: pathlib.Path):
    import io

    from langchain_core.runnables.graph import CurveStyle
    from PIL import Image

    graph = create_tree_based_experimentation_pipeline()

    image = Image.open(
        io.BytesIO(graph.get_graph().draw_mermaid_png(curve_style=CurveStyle.LINEAR)),
    )
    image.save(root_dir / ".github" / "assets" / "graph-tree-based-experimentation.png")


def test_run_tree_based_experimentation_pipeline(
    idea: Dict[str, str], code_to_use: str
):
    output = run_tree_based_experimentation_pipeline(
        idea=idea,
        code_to_use=code_to_use,
    )
    breakpoint()
