from typing import Annotated, Dict, Final, List, Literal, Optional, Union

from langchain.chat_models import init_chat_model
from langchain_community.tools.semanticscholar import SemanticScholarQueryRun
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import add_messages
from langgraph.prebuilt import create_react_agent
from loguru import logger
from pydantic import BaseModel

from ai_scientist_v2.models import Idea
from ai_scientist_v2.utils.configuration import Configuration
from ai_scientist_v2.utils.stream import print_stream

from .utils import get_last_tool_result, jsonl_to_prompt_string

SYSTEM_PROMPT: Final[str] = """\
You are an experienced AI researcher who aims to propose high-impact research ideas resembling exciting grant proposals. Feel free to propose any novel ideas or experiments; make sure they are novel. Be very creative and think out of the box. Each proposal should stem from a simple and elegant question, observation, or hypothesis about the topic. For example, they could involve very interesting and simple interventions or investigations that explore new possibilities or challenge existing assumptions. Clearly clarify how the proposal distinguishes from the existing literature.

Ensure that the proposal does not require resources beyond what an academic lab could afford. These proposals should lead to papers that are publishable at top ML conferences.

If you decide to finalize your idea, please output your idea in the following format:

```
# Idea
## Name
A short descriptor of the idea, lowercase, no spaces, underscores allowed.

## Title
A catchy and informative title for the proposal.

## Short Hypothesis
A concise statement of the main hypothesis or research question. Clarify the need for this specific direction, ensure this is the best setting to investigate this idea, and there are not obvious other simpler ways to answer the question.

## Related Work
A brief discussion of the most relevant related work and how the proposal clearly distinguishes from it, and is not a trivial extension.

## Abstract
An abstract that summarizes the proposal in conference format (approximately 250 words).

## Experiments
- A list of experiments that would be conducted to validate the proposal. Ensure these are simple and feasible. Be specific in exactly how you would test the hypothesis, and detail precise algorithmic changes. Include the evaluation metrics you would use.

## Risk Factors and Limitations
- A list of potential risks and limitations of the proposal.
```

Note: You should perform at least one literature search using the semantic scholar search before finalizing your idea to ensure it is well-informed by existing research."""


IDEA_GENERATION_PROMPT: Final[str] = """\
{workshop_description}

Here are the proposals that you have already generated:

'''
{prev_ideas}
'''

Begin by generating an interestingly new high-level research proposal that diffsers from what you have previously proposed."""

IDEA_REFLECTION_PROMPT: Final[str] = """\
Round {current_round} / {max_reflections}.

In your thoughts, first carefully consider the quality, novelty, and feasibility of the proposal you just created.
Include any other factors that you think are important in evaluating the proposal.
Ensure the proposal is clear and concise, and the JSON is in the correct format.
Do not make things overly complicated.
In the next attempt, try to refine and improve your proposal.
Stick to the spirit of the original idea unless there are glaring issues.

If you have new information from tools, such as literature search results, incorporate them into your reflection and refine your proposal accordingly.

Results from your last action (if any):

{last_tool_result}"""

FINALIZE_PROMPT: Final[str] = """\
Finalize your idea by provided the idea details. 

If you finalize your idea, please output your idea in the following format.

```json
{{
    "name": "...",
    "title": "...",
    "short_hypothesis": "...",
    "related_work": "...",
    "abstract": "...",
    "experiments": ["...", "...", ...],
    "risk_factors_and_limitations": ["...", "...", ...]
}}
```

Ensure the JSON is properly formatted for automatic parsing."""


class IdeaGenerationState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    current_round: int
    last_tool_result: Optional[str] = None
    generated_idea: Optional[Idea] = None


class IdeaGenerationConfig(Configuration):
    idea_generation_model_name: str
    idea_reflection_model_name: str

    workshop_description: str
    prev_ideas: List[Dict[str, Union[str, List[str]]]]
    max_reflections: int

    @property
    def prev_ideas_for_prompt(self) -> str:
        return jsonl_to_prompt_string(self.prev_ideas)


def idea_generation_node(
    state: IdeaGenerationState, config: RunnableConfig
) -> IdeaGenerationState:
    conf = IdeaGenerationConfig.from_runnable_config(config)

    agent = create_react_agent(
        model=init_chat_model(model=conf.idea_generation_model_name),
        tools=[SemanticScholarQueryRun()],
    )
    system_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT)
    human_prompt = HumanMessagePromptTemplate.from_template(IDEA_GENERATION_PROMPT)

    prompt = ChatPromptTemplate.from_messages(messages=[system_prompt, human_prompt])
    chain = prompt | agent

    input_dict = {
        "workshop_description": conf.workshop_description,
        "prev_ideas": conf.prev_ideas_for_prompt,
    }
    output = print_stream(stream=chain.stream(input_dict))
    messages = output["messages"][1:]  # messages[0] is the system prompt so we skip it

    last_tool_result = get_last_tool_result(messages)

    return IdeaGenerationState(
        messages=messages,
        last_tool_result=last_tool_result,
        current_round=state.current_round,
    )


def idea_reflection_node(
    state: IdeaGenerationState, config: RunnableConfig
) -> IdeaGenerationState:
    conf = IdeaGenerationConfig.from_runnable_config(config)

    agent = create_react_agent(
        model=init_chat_model(model=conf.idea_reflection_model_name),
        tools=[SemanticScholarQueryRun()],
    )
    system_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT)
    human_prompt = HumanMessagePromptTemplate.from_template(IDEA_REFLECTION_PROMPT)

    prompt = ChatPromptTemplate.from_messages(
        messages=[system_prompt] + state.messages + [human_prompt]
    )
    chain = prompt | agent

    input_dict = {
        "current_round": state.current_round + 1,
        "max_reflections": conf.max_reflections,
        "last_tool_result": state.last_tool_result,
    }
    output = print_stream(stream=chain.stream(input_dict))
    messages = output["messages"][1:]  # messages[0] is the system prompt so we skip it

    return IdeaGenerationState(
        messages=messages,
        current_round=state.current_round + 1,
        last_tool_result=state.last_tool_result,
    )


def idea_finalization_node(
    state: IdeaGenerationState, config: RunnableConfig
) -> IdeaGenerationState:
    conf = IdeaGenerationConfig.from_runnable_config(config)

    agent = init_chat_model(
        model=conf.idea_generation_model_name,
    )

    system_prompt = SystemMessage(SYSTEM_PROMPT)
    human_prompt = HumanMessage(FINALIZE_PROMPT)
    messages = [system_prompt] + state.messages + [human_prompt]

    agent_with_structured_output = agent.with_structured_output(Idea)
    generated_idea = agent_with_structured_output.invoke(messages)
    assert isinstance(generated_idea, Idea)

    last_tool_result = get_last_tool_result(messages)

    return IdeaGenerationState(
        messages=messages,
        current_round=state.current_round,
        last_tool_result=last_tool_result,
        generated_idea=generated_idea,
    )


def should_continue(
    state: IdeaGenerationState, config: RunnableConfig
) -> Literal["idea-reflection", "idea-finalization"]:
    conf = IdeaGenerationConfig.from_runnable_config(config)

    if state.current_round >= conf.max_reflections:
        # End after max_reflections.
        logger.info("Max reflections reached, finalizing idea.")
        return "idea-finalization"
    else:
        return "idea-reflection"
