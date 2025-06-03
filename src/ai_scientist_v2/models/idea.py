from typing import List

from pydantic import BaseModel, Field


class Idea(BaseModel):
    name: str = Field(
        description="A short descriptor of the idea. Lowercase, no spaces, underscores allowed.",
    )
    title: str = Field(
        description="A catchy and informative title for the proposal.",
    )
    short_hypothesis: str = Field(
        description="A concise statement of the main hypothesis or research question. Clarify the need for this specific direction, ensure this is the best setting to investigate this idea, and there are not obvious other simpler ways to answer the question.",
    )
    related_work: str = Field(
        description="A brief discussion of the most relevant related work and how the proposal clearly distinguishes from it, and is not a trivial extension.",
    )
    abstract: str = Field(
        description="An abstract that summarizes the proposal in conference format (approximately 250 words).",
    )
    experiments: List[str] = Field(
        description="A list of experiments that would be conducted to validate the proposal. Ensure these are simple and feasible. Be specific in exactly how you would test the hypothesis, and detail precise algorithmic changes. Include the evaluation metrics you would use.",
    )
    risk_factors_and_limitations: List[str] = Field(
        description="A list of potential risks and limitations of the proposal.",
    )
