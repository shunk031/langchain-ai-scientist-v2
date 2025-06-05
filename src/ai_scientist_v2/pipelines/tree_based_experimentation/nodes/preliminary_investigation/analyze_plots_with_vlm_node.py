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
You are an experienced AI researcher analyzing experimental results. You have been provided with plots from a machine learning experiment. This experiment is based on the following research idea: 

{research_idea}
"""

ANALYZE_PLOT_PROMPT: Final[str] = """\
Please analyze these plots and provide detailed insights about the results. If you don't receive any plots, say 'No plots received'. Never make up plot analysis. Please return the analyzes with strict order of uploaded images, but DO NOT include any word like 'the first plot'.
"""


def analyze_plots_with_vlm_node(state, config):
    raise NotImplementedError
