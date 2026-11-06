"""CPU coverage for the CubePart RoPE call and the PartSAM motion review.

Test inventory:
- test_qwen_embed_rope_legacy_call_raises_and_fixed_call_matches_signature
- test_motion_review_crash_is_fail_and_clean_verdicts_are_recorded
- test_ledger_records_cubepart_mesh_urdf_and_review
- test_default_skips_cubepart_when_stored_flag_is_false
- test_forced_sf10_dc1_runs_cubepart_and_records_the_stored_flag
- test_urdf_provenance_accepts_a_partsam_urdf
- test_urdf_provenance_rejects_partnet_25959
- test_urdf_provenance_rejects_link_count_mismatch
- test_urdf_provenance_rejects_unmatched_mesh
- test_diffusers_version_0_39_0_passes_and_other_versions_raise
"""

import hashlib
import json
import logging
from pathlib import Path

import pytest

from sceneagent.section34 import (
    DiffusersVersionError,
    call_qwen_embed_rope,
    run_cubepart,
    run_partsam_articulation,
)


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


def _xml_attr(value):
    return str(value).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


def _mesh_parts(directory, names=("cap", "barrel")):
    directory.mkdir(parents=True, exist_ok=True)
    parts = []
    for name in names:
        mesh = directory / f"{name}.obj"
        mesh.write_bytes(f"solid {name}\n".encode())
        parts.append({"name": name, "mesh": str(mesh)})
    return parts


def _link_xml(part):
    name = part["name"]
    mesh = part.get("mesh")
    if not mesh:
        return f'  <link name="{name}"/>'
    return (
        f'  <link name="{name}">\n'
        f'    <visual><geometry><mesh filename="{_xml_attr(mesh)}"/></geometry></visual>\n'
        "  </link>"
    )


def _write_part_urdf(call):
    spec = json.loads(Path(call["spec_path"]).read_text(encoding="utf-8"))
    parts = spec["parts"]
    links = "\n".join(_link_xml(part) for part in parts)
    joint = ""
    if len(parts) >= 2:
        parent = parts[0]["name"]
        child = parts[1]["name"]
        joint = (
            f'  <joint name="{parent}_to_{child}" type="revolute">\n'
            f'    <parent link="{parent}"/>\n'
            f'    <child link="{child}"/>\n'
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
    parts = _mesh_parts(tmp_path / "meshes")
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
        _mesh_parts(tmp_path / "link_meshes"),
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


def _accept_review(_call):
    return _Proc(0, '{"failure_reason": "success", "realism_rating": 10}', "")


def _propose_text(text):
    def propose(call):
        Path(call["urdf_path"]).write_text(text, encoding="utf-8")
        return _Proc(0)

    return propose


def _two_part_assets(directory):
    cap = directory / "cap.obj"
    barrel = directory / "barrel.obj"
    cap.write_bytes(b"cap-bytes")
    barrel.write_bytes(b"barrel-bytes")
    parts = [
        {"name": "cap", "mesh": str(cap)},
        {"name": "barrel", "sha256": hashlib.sha256(b"barrel-bytes").hexdigest()},
    ]
    urdf = (
        '<robot name="partsam">\n'
        f'  <link name="cap"><visual><geometry><mesh filename="{cap}"/></geometry></visual></link>\n'
        f'  <link name="barrel"><visual><geometry><mesh filename="{barrel}"/></geometry></visual></link>\n'
        "</robot>\n"
    )
    return parts, urdf


def test_default_skips_cubepart_when_stored_flag_is_false(tmp_path):
    calls = []

    def boom():
        calls.append("called")
        raise AssertionError("the default path must not run CubePart")

    skipped = run_cubepart(tmp_path / "skip", False, infer=boom, version_lookup=boom)
    assert calls == []
    assert skipped["ran"] is False
    assert skipped["completion_needed"] is False
    entry = _ledger_entry(tmp_path / "skip", "cubepart")
    assert entry["decision"] == "reject"
    assert entry["completion_needed"] is False
    assert entry["forced_for"] is None
    assert "model_reply" not in entry
    other = run_cubepart(
        tmp_path / "other", False, force=True, infer=boom, version_lookup=boom
    )
    assert other["ran"] is False
    assert calls == []


def test_forced_sf10_dc1_runs_cubepart_and_records_the_stored_flag(tmp_path):
    mesh = tmp_path / "forced.glb"
    calls = []

    def infer():
        calls.append("ran")
        mesh.write_bytes(b"forced-mesh")
        return mesh

    forced = run_cubepart(
        tmp_path / "force",
        False,
        force="SF-10 dc1",
        infer=infer,
        version_lookup=lambda: "0.39.0",
    )
    assert calls == ["ran"]
    assert forced["ran"] is True
    assert forced["completion_needed"] is False
    assert forced["forced_for"] == "SF-10 dc1"
    entry = _ledger_entry(tmp_path / "force", "cubepart")
    assert entry["decision"] == "accept"
    assert entry["forced_for"] == "SF-10 dc1"
    assert entry["completion_needed"] is False
    assert entry["diffusers_version"] == "0.39.0"
    assert entry["part_mesh"] == str(mesh)
    assert "model_reply" not in entry


def test_urdf_provenance_accepts_a_partsam_urdf(tmp_path):
    parts, urdf = _two_part_assets(tmp_path)
    out = tmp_path / "good"
    run_partsam_articulation(
        out, parts, propose=_propose_text(urdf), review=_accept_review
    )
    entry = _ledger_entry(out, "articulate")
    assert entry["decision"] == "accept"
    assert entry["review"] == "accept"
    assert entry["links_match"] is True


def test_urdf_provenance_rejects_partnet_25959(tmp_path):
    parts, urdf = _two_part_assets(tmp_path)
    tainted = urdf.replace(
        "<robot",
        "<robot><!-- partnet-mobility object 25959 -->",
        1,
    )
    out = tmp_path / "partnet"
    run_partsam_articulation(
        out, parts, propose=_propose_text(tainted), review=_accept_review
    )
    entry = _ledger_entry(out, "articulate")
    assert entry["decision"] == "reject"
    assert entry["review"] == "reject"
    assert "25959" in entry["reason"]
    assert "PartNet" in entry["reason"]


def test_urdf_provenance_rejects_link_count_mismatch(tmp_path):
    parts, urdf = _two_part_assets(tmp_path)
    cap_only = "\n".join(
        line for line in urdf.splitlines(keepends=True) if "barrel" not in line
    )
    out = tmp_path / "count"
    run_partsam_articulation(
        out, parts, propose=_propose_text(cap_only), review=_accept_review
    )
    entry = _ledger_entry(out, "articulate")
    assert entry["decision"] == "reject"
    assert "link count 1" in entry["reason"]
    assert "part count 2" in entry["reason"]


def test_urdf_provenance_rejects_unmatched_mesh(tmp_path):
    parts, urdf = _two_part_assets(tmp_path)
    other = tmp_path / "other.obj"
    other.write_bytes(b"not-the-barrel")
    swapped = urdf.replace(str(tmp_path / "barrel.obj"), str(other))
    out = tmp_path / "mesh"
    run_partsam_articulation(
        out, parts, propose=_propose_text(swapped), review=_accept_review
    )
    entry = _ledger_entry(out, "articulate")
    assert entry["decision"] == "reject"
    assert "does not match a PartSAM part" in entry["reason"]


def test_diffusers_version_0_39_0_passes_and_other_versions_raise(tmp_path, caplog):
    from sceneagent.section34 import require_diffusers_039

    with caplog.at_level(logging.INFO, logger="sceneagent.section34"):
        assert require_diffusers_039(lambda: "0.39.0") == "0.39.0"
    assert "installed diffusers version 0.39.0" in caplog.text
    for bad in ("0.38.0", "0.38.2", "0.40", "0.40.0"):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="sceneagent.section34"):
            with pytest.raises(DiffusersVersionError):
                require_diffusers_039(lambda bad=bad: bad)
        assert f"installed diffusers version {bad}" in caplog.text
    calls = []
    with pytest.raises(DiffusersVersionError):
        run_cubepart(
            tmp_path / "bad",
            False,
            force="SF-10 dc1",
            infer=lambda: calls.append("ran"),
            version_lookup=lambda: "0.38.1",
        )
    assert calls == []
