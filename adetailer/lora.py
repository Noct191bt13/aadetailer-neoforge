from __future__ import annotations

import re
from typing import NamedTuple


class LoraInfo(NamedTuple):
    token: str
    triggers: tuple[str, ...]


# Legacy tag matcher: any <name:float> extra-network tag.
LORA_RE = re.compile(r"<([^<>]*):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*>")
# Full <lora:...> tokens, including loractl schedule/named syntax like
# <lora:Name:0@0; 0.5@8:hr=0.7:ad=0.5> (LORA_RE misses those because the weight
# slot is not a plain float).
LORA_TOKEN_RE = re.compile(r"<lora:([^<>]+)>", re.IGNORECASE)
LORA_TRIGGER_RE = re.compile(r"\(([^()]+)\)")


def extract_lora_triggers(name: str) -> tuple[str, ...]:
    """Return the parenthesized trigger words embedded in a lora name.

    Trigger convention (Noct191bt13 fork): the part inside parentheses in the
    lora filename is treated as a trigger tag, e.g. "<lora:Name (trigger):1>".
    """
    triggers: list[str] = []
    for match in LORA_TRIGGER_RE.finditer(name):
        trigger = match.group(1).strip()
        if trigger and trigger not in triggers:
            triggers.append(trigger)
    return tuple(triggers)


def find_loras(prompt: str) -> list[LoraInfo]:
    """Find every lora token in a prompt, with its embedded trigger words.

    Both plain <lora:Name:0.8> tags and loractl-style scheduled tokens
    (<lora:Name:0@0; 0.5@8:hr=0.7:ad=0.5>) are matched; bare <name:float>
    tags of other extra networks are kept for backward compatibility.
    """
    if not prompt:
        return []
    loras: list[LoraInfo] = []
    seen_tokens = set()
    for match in LORA_TOKEN_RE.finditer(prompt):
        token = match.group(0)
        if token in seen_tokens:
            continue
        seen_tokens.add(token)
        name = match.group(1).split(":")[0].strip()
        triggers = extract_lora_triggers(name)
        loras.append(LoraInfo(token=token, triggers=triggers))
    for match in LORA_RE.finditer(prompt):
        token = match.group(0)
        if token in seen_tokens:
            continue
        seen_tokens.add(token)
        name = match.group(1).strip()
        triggers = extract_lora_triggers(name)
        loras.append(LoraInfo(token=token, triggers=triggers))
    return loras


def strip_lora_tokens(prompt: str) -> str:
    """Remove lora tokens from a prompt (used for trigger-usage checks)."""
    return LORA_TOKEN_RE.sub("", LORA_RE.sub("", prompt))
