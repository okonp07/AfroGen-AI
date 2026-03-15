from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptProfile:
    age_group: str
    skin_tone: str
    hairstyle: str
    expression: str
    accessory: str


def _contains(prompt: str, options: tuple[str, ...]) -> bool:
    return any(option in prompt for option in options)


def parse_prompt(prompt: str) -> PromptProfile:
    lowered = prompt.lower()

    if _contains(lowered, ("child", "kid", "young girl", "young boy")):
        age_group = "child"
    elif _contains(lowered, ("elder", "older", "old", "senior")):
        age_group = "senior"
    else:
        age_group = "adult"

    if _contains(lowered, ("deep", "dark", "ebony", "rich dark")):
        skin_tone = "deep"
    elif _contains(lowered, ("light", "warm brown", "golden")):
        skin_tone = "warm"
    else:
        skin_tone = "medium"

    if _contains(lowered, ("braid", "braids", "cornrow", "cornrows")):
        hairstyle = "braids"
    elif _contains(lowered, ("afro", "rounded hair")):
        hairstyle = "afro"
    elif _contains(lowered, ("loc", "dread", "twist")):
        hairstyle = "locs"
    else:
        hairstyle = "curly"

    if _contains(lowered, ("serious", "calm", "stoic")):
        expression = "calm"
    elif _contains(lowered, ("smile", "smiling", "joy", "happy")):
        expression = "smile"
    else:
        expression = "neutral"

    if _contains(lowered, ("glasses", "spectacles")):
        accessory = "glasses"
    elif _contains(lowered, ("earrings", "hoops")):
        accessory = "earrings"
    else:
        accessory = "none"

    return PromptProfile(
        age_group=age_group,
        skin_tone=skin_tone,
        hairstyle=hairstyle,
        expression=expression,
        accessory=accessory,
    )
