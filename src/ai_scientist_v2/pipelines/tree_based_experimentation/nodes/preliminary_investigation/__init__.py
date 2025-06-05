from ai_scientist_v2.models import Idea
from ai_scientist_v2.utils.configuration import Configuration

from .analyze_plots_with_vlm_node import analyze_plots_with_vlm_node
from .create_draft_node import create_draft_node
from .debug_code_node import debug_code_node
from .define_global_metrics import define_global_metrics_node
from .extract_metrics import extract_metrics_node
from .generate_ablation import generate_ablation_node
from .generate_plotting_code import generate_plotting_code_node
from .generate_substage_goal import generate_substage_goal_node
from .parse_exec_result import parse_exec_result_node
from .select_generated_plots import select_generated_plots


class AgentConfig(Configuration):
    """Configuration for the agent used in the preliminary investigation node."""

    idea: Idea


__all__ = [
    "AgentConfig",
    "analyze_plots_with_vlm_node",
    "create_draft_node",
    "debug_code_node",
    "define_global_metrics_node",
    "extract_metrics_node",
    "generate_ablation_node",
    "generate_plotting_code_node",
    "generate_substage_goal_node",
    "parse_exec_result_node",
    "select_generated_plots",
]
