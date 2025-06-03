from typing import Final

SYSTEM_PROMPT: Final[str] = """\
You are an experienced AI researcher. You are provided with a previously developed baseline implementation. Your task is to implement hyperparameter tuning for the following idea:

{hyperprameter_idea_name}

{hyperparameter_idea_description}

Base code you are working on:

{base_code}

Instructions:
"""
