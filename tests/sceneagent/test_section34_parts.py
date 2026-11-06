"""CPU coverage for the CubePart RoPE call and the PartSAM motion review."""

import json
from pathlib import Path

from sceneagent.section34 import call_qwen_embed_rope, run_partsam_articulation


class _Proc:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _Diffusers039:
    """diffusers 0.39: the second parameter is device, then max_txt_seq_len."""

    def forward(self, video_fhw, device=None, max_txt_seq_len=None):
        self.video_fhw = video_fhw
        self.device = device
        self.max_txt_seq_len = max_txt_seq_len
        return ("vid", "txt")


class _RepoSignature:
    """The argument names the CubePart repo already passes."""

    def forward(self, img_shapes, txt_seq_lens, device):
        self.img_shapes = img_shapes
        self.txt_seq_lens = txt_seq_lens
        self.device = device
        return ("vid", "txt")


def _write_part_urdf(call):
    spec = json.loads(Path(call["spec_path"]).read_text(encoding="utf-8"))
    names = [part["name"] for part in spec["parts"]]
    links = "\n".join(f'  <link name="{name}"/>' for name in names)
    joint = ""
    if len(names) >= 2:
        joint = (
            f'  <joint name="{names[0]}_to_{names[1]}" type="revolute">\n'
            f'    <parent link="{names[0]}"/>\n'
            f'    <child link="{names[1]}"/>\n'
            '    <limit lower="0" upper="1"/>\n'
            "  </joint>\n"
        )
    Path(call["urdf_path"]).write_text(
        f'<robot name="partsam">\n{links}\n{joint}</robot>\n',
        encoding="utf-8",
    )
    return _Proc(0)


def _ledger_entry(out, item):
    payload = json.loads((Path(out) / "ledger.json").read_text(encoding="utf-8"))
    matches = [entry for entry in payload["entries"] if entry["item"] == item]
    assert matches, item
    return matches[-1]


def test_qwen_embed_rope_legacy_call_raises_and_fixed_call_matches_signature():
    shapes = [(1, 8, 8)]
    lengths = [4, 12, 7]
    device = "cpu"
    embed = _Diffusers039()
    original = embed.forward.__func__
    # CubePart passes the text length in the slot diffusers 0.39 named device.
    try:
        embed.forward(shapes, lengths, device=device)
    except TypeError as exc:
        message = str(exc)
    else:
        raise AssertionError("the CubePart call should raise on diffusers 0.39")
    assert "multiple values for argument 'device'" in message

    produced = call_qwen_embed_rope(embed, shapes, lengths, device)
    assert produced == ("vid", "txt")
    assert embed.video_fhw == shapes
    assert embed.device == device
    assert embed.max_txt_seq_len == 12
    assert embed.forward.__func__ is original

    repo = _RepoSignature()
    call_qwen_embed_rope(repo, shapes, lengths, device)
    assert repo.img_shapes == shapes
    assert repo.txt_seq_lens == lengths
    assert repo.device == device
    assert repo.forward.__func__ is _RepoSignature.forward


def test_motion_review_crash_is_fail_and_clean_verdicts_are_recorded(tmp_path):
    parts = [{"name": "cap"}, {"name": "barrel"}]
    cases = (
        (139, '{"failure_reason": "success", "realism_rating": 10}', "fail"),
        (-11, '{"decision": "accept"}', "fail"),
        (0, '{"failure_reason": "success", "realism_rating": 10}', "accept"),
        (0, '{"failure_reason": "joint_axis", "feedback_score": -1}', "reject"),
    )
    for code, text, expected in cases:
        out = tmp_path / f"review_{code}"

        def review(call, code=code, text=text):
            assert "articulate_anything/physics/sapien_simulate.py" in call["command"]
            assert any("mobility.urdf" in part for part in call["command"])
            return _Proc(code, text, "")

        body = run_partsam_articulation(
            out, parts, propose=_write_part_urdf, review=review
        )
        entry = _ledger_entry(out, "articulate")
        assert entry["review"] == expected
        assert entry["decision"] == expected
        assert entry["returncode"] == code
        assert entry["urdf"] == body["urdf"]
        assert Path(entry["urdf"]).is_file()
        if expected == "fail":
            assert entry["decision"] != "accept"
            assert entry["review"] != "accept"

    table = tmp_path / "table"

    def table_urdf(call):
        Path(call["urdf_path"]).write_text(
            '<robot name="table"><link name="base"/><link name="wheel"/></robot>\n',
            encoding="utf-8",
        )
        return _Proc(0)

    def table_review(_call):
        return _Proc(0, '{"failure_reason": "success", "realism_rating": 10}', "")

    run_partsam_articulation(table, parts, propose=table_urdf, review=table_review)
    rejected = _ledger_entry(table, "articulate")
    assert rejected["decision"] == "reject"
    assert rejected["review"] == "reject"
    assert rejected["links_match"] is False
    assert "PartSAM" in rejected["reason"]


def test_ledger_records_cubepart_mesh_urdf_and_review(tmp_path):
    mesh = tmp_path / "0001_parts.glb"
    mesh.write_bytes(b"part-mesh")
    out = tmp_path / "run"
    held = run_partsam_articulation(out / "held", ["cap"])
    assert held["review"] == "reject"
    assert held["entries"][-1]["decision"] == "reject"
    assert "not started" in held["entries"][-1]["reason"]

    body = run_partsam_articulation(
        out,
        [{"name": "cap"}, {"name": "barrel"}],
        part_mesh=mesh,
        propose=_write_part_urdf,
        review=lambda _call: _Proc(
            0, '{"failure_reason": "success", "realism_rating": 10}', ""
        ),
    )
    ledger = json.loads((out / "ledger.json").read_text(encoding="utf-8"))
    cubepart = next(entry for entry in ledger["entries"] if entry["item"] == "cubepart")
    articulate = next(
        entry for entry in ledger["entries"] if entry["item"] == "articulate"
    )
    assert cubepart["part_mesh"] == str(mesh)
    assert cubepart["artifact"] == str(mesh)
    assert cubepart["decision"] == "accept"
    assert Path(cubepart["part_mesh"]).read_bytes() == b"part-mesh"
    assert articulate["part_mesh"] == str(mesh)
    assert articulate["urdf"] == body["urdf"]
    assert articulate["urdf"].endswith("mobility.urdf")
    assert Path(articulate["urdf"]).is_file()
    assert articulate["review"] == "accept"
    assert articulate["decision"] == "accept"
    assert articulate["links_match"] is True
    manifest = json.loads(
        (out / "manifests" / "articulate-1.json").read_text(encoding="utf-8")
    )
    assert manifest["detail"]["review_command"][1].endswith("sapien_simulate.py")
    assert manifest["detail"]["propose_command"][1] == "articulate.py"
