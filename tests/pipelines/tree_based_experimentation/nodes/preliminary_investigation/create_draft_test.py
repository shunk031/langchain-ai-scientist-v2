import json
import pathlib

import pytest
from langchain_core.runnables import RunnableConfig

from ai_scientist_v2.config import BFTSConfig
from ai_scientist_v2.pipelines.tree_based_experimentation import (
    TreeBasedExperimentationState,
)
from ai_scientist_v2.pipelines.tree_based_experimentation.nodes.preliminary_investigation import (
    create_draft_node,
)


@pytest.fixture
def state(test_fixtures_dir: pathlib.Path) -> TreeBasedExperimentationState:
    json_file = (
        test_fixtures_dir
        / "pipelines"
        / "tree_based_experimentation"
        / "nodes"
        / "preliminary_investigation"
        / "output_define_global_metrics_node.json"
    )
    with json_file.open("r") as rf:
        state_data = json.load(rf)

    return TreeBasedExperimentationState.model_validate(state_data)


@pytest.mark.parametrize(
    argnames="num_syn_datasets",
    argvalues=[1, 2, 3],
)
@pytest.mark.parametrize(
    argnames="k_fold_validation",
    argvalues=[1, 5, 10],
)
def test_create_draft_node(
    state: TreeBasedExperimentationState,
    num_syn_datasets: int,
    k_fold_validation: int,
):
    bfts_config = BFTSConfig()
    bfts_config.experiment.num_syn_datasets = num_syn_datasets
    bfts_config.agent.k_fold_validation = k_fold_validation

    config = RunnableConfig(
        {
            "configurable": {
                "hoge": "fuga",
            }
        }
    )
    output = create_draft_node(state=state, config=config)
