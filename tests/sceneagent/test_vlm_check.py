# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Advisory verdict parsing. The check does not change a gate."""

from __future__ import annotations

from sceneagent.vlm_check import advise, verdict_of

# The reply from articulate run 43468919336.
MARKDOWN_YES = (
    "**Yes**, the expected joint motion is visible. Across the frames, the "
    "top part of the cube acts as a hinged lid, rotating upward around a "
    "revolute joint located along the rear upper edge."
)


class _Stub:
    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[str] = []

    def generate_images(self, prompt, images=(), search=False):
        self.prompts.append(prompt)
        return {"text": self.text, "model_id": "stub-model", "http_status": 200}


def test_the_articulate_markdown_yes_parses_and_stays_advisory():
    stub = _Stub(MARKDOWN_YES)
    report = advise([b"a", b"b"], "Is the expected joint motion visible?", stub)
    assert report["verdict"] == "yes"
    assert report["advisory"] is True
    assert report["label"] == "vlm_check"
    assert "status" not in report
    assert '"verdict"' in stub.prompts[0]


def test_markdown_and_punctuation_around_no():
    assert verdict_of("**No**, the lid stays shut.") == "no"
    assert verdict_of("No - the parts do not move.") == "no"
    assert verdict_of('"No -"') == "no"
    assert verdict_of("Yes.") == "yes"


def test_a_json_verdict_wins():
    assert verdict_of('{"verdict": "no", "reason": "The lid stays shut."}') == "no"
    assert (
        verdict_of('```json\n{"verdict": "Yes.", "reason": "It opens."}\n```') == "yes"
    )


def test_ambiguous_replies_stay_unparsed():
    assert verdict_of("The frames do not show a clear hinge.") == "unparsed"
    assert verdict_of("Maybe the lid opens.") == "unparsed"
    assert verdict_of("Yesterday the lid was shut.") == "unparsed"
    assert verdict_of("Notice the cube.") == "unparsed"
    assert verdict_of('{"verdict": "maybe", "reason": "Unclear."}') == "unparsed"
