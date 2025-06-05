import pathlib
from typing import Dict, List

import pytest

from ai_scientist_v2.pipelines import (
    create_tree_based_experimentation_pipeline,
    run_tree_based_experimentation_pipeline,
)


@pytest.fixture
def code_to_use() -> str:
    return "No code is provided for this stage."


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
