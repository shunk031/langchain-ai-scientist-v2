from ai_scientist_v2.config import BFTSAgentConfig
from ai_scientist_v2.models import Idea
from ai_scientist_v2.utils.configuration import Configuration

from .define_global_metrics import define_global_metrics_node
from .generate_substage_goal import generate_substage_goal_node


class AgentConfig(BFTSAgentConfig, Configuration):
    """Configuration for the agent used in the preliminary investigation node."""

    idea: Idea


__all__ = [
    "AgentConfig",
    "define_global_metrics_node",
    "generate_substage_goal_node",
]
