from __future__ import annotations

from adetailer.lora import (
    LORA_RE,
    LORA_TOKEN_RE,
    extract_lora_triggers,
    find_loras,
    strip_lora_tokens,
)

SCHEDULED = "<lora:NoctAnimaV7:0@0; 0.5@8; 0.7@20:hr=0.7:ad=0.5>"
PLAIN = "<lora:NoctAnimaV7:0.8>"
PURE_SCHEDULE = "<lora:Other:0@0,1@5>"
WITH_TRIGGER = "<lora:NoctAnimaV7 (nana):0.8>"


def test_find_loras_plain() -> None:
    loras = find_loras(f"masterpiece, {PLAIN}, 1girl")
    assert [l.token for l in loras] == [PLAIN]
    assert loras[0].triggers == ()


def test_find_loras_scheduled_token() -> None:
    loras = find_loras(f"masterpiece, {SCHEDULED}, 1girl")
    assert [l.token for l in loras] == [SCHEDULED]
    assert loras[0].triggers == ()


def test_find_loras_pure_schedule_without_named_params() -> None:
    loras = find_loras(f"masterpiece, {PURE_SCHEDULE}, 1girl")
    assert [l.token for l in loras] == [PURE_SCHEDULE]


def test_find_loras_triggers() -> None:
    loras = find_loras(f"{WITH_TRIGGER}")
    assert loras[0].token == WITH_TRIGGER
    assert loras[0].triggers == ("nana",)


def test_find_loras_deduplicates() -> None:
    loras = find_loras(f"{PLAIN}, {PLAIN}")
    assert len(loras) == 1


def test_find_loras_legacy_non_lora_tag() -> None:
    loras = find_loras("<hypernet:style:1.0>")
    assert [l.token for l in loras] == ["<hypernet:style:1.0>"]


def test_find_loras_empty() -> None:
    assert find_loras("") == []
    assert find_loras("no loras here") == []


def test_extract_lora_triggers() -> None:
    assert extract_lora_triggers("Name (alpha) (beta)") == ("alpha", "beta")
    assert extract_lora_triggers("Name") == ()


def test_strip_lora_tokens() -> None:
    prompt = f"a, {PLAIN}, b, {SCHEDULED}, c"
    stripped = strip_lora_tokens(prompt)
    assert PLAIN not in stripped and SCHEDULED not in stripped
    assert "a" in stripped and "b" in stripped and "c" in stripped


def test_regexes_match_schedule_token() -> None:
    assert LORA_TOKEN_RE.search(SCHEDULED) is not None
    # the legacy LORA_RE does NOT match scheduled tokens: every float is
    # prefixed by a name= (e.g. :hr=0.7, :ad=0.5) or sits after a ;/@
    assert LORA_RE.search(SCHEDULED) is None
    assert LORA_RE.search(PURE_SCHEDULE) is None
