# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The gpu-smoke `setup` pipeline: box stack install and the Hugging Face token.

No test reaches a box or GitHub. SSH is replaced by a fake subprocess.run.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from sceneagent import gpu_smoke_trigger as gst
from sceneagent import vast_smoke as vs

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "sceneagent-gpu-smoke.yml"
SCRIPT = ROOT / "sceneagent" / "box_setup.sh"
TOKEN = "hf_" + "x" * 34


def test_the_setup_script_parses_and_names_every_stage():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    text = SCRIPT.read_text(encoding="utf-8")
    stages = text.split('STAGES="', 1)[1].split('"', 1)[0].split()
    assert stages[0] == "apt" and stages[-1] == "isaaclab"
    assert stages[-2] == "sam3d_weights"
    for name in stages:
        assert f"s_{name}() {{" in text


def test_setup_writes_the_paths_the_pipeline_code_reads():
    from sceneagent import bin_masks, bin_sam3d, bin_scale, bin_splat
    from sceneagent.simfoundry_python import SIMFOUNDRY_PYTHON

    text = SCRIPT.read_text(encoding="utf-8")
    assert SIMFOUNDRY_PYTHON == "/root/.local/share/mamba/envs/sa/bin/python3"
    assert "ENVS=/root/.local/share/mamba/envs" in text
    assert bin_splat.NS_BIN.endswith("/envs/nerf/bin")
    assert bin_masks.MASK_PYTHON.endswith("/envs/gsam2/bin/python3")
    assert f"GSAM_ROOT={bin_masks.GSAM_ROOT}" in text
    assert f"SAM_WEIGHTS={bin_masks.SAM_WEIGHTS}" in text
    assert f"SAM3D_ROOT={bin_sam3d.SAM3D_ROOT}" in text
    assert f"SAM3D_ENV={bin_sam3d.SAM3D_ENV}" in text
    assert f"DA3_ROOT={bin_scale.DA3_SRC.removesuffix('/src')}" in text
    assert "/opt/colmap-cuda" in text and vs.REMOTE_COLMAP_BIN == "/opt/colmap-cuda/bin"


def test_the_report_never_carries_the_token():
    text = SCRIPT.read_text(encoding="utf-8")
    report = text.split("do_report()", 1)[1]
    assert "read_text" not in report.split("hf_token_present", 1)[1].split("\n")[0]
    assert "SECRETISH" in report  # log tails drop token-looking lines


def test_setup_runs_detached_and_then_reports():
    script = vs.build_remote_shell(
        vs.REMOTE_REPO, vs.remote_out("setup"), pipeline="setup"
    )
    assert vs.remote_out("setup") == vs.REMOTE_SETUP_OUT
    assert "setsid nohup bash" in script and " run " in script
    assert script.rstrip().endswith(f"report {vs.REMOTE_SETUP_OUT}")
    assert "token" not in script.lower()
    assert "setup" in vs.PIPELINES and vs.PIPELINES["setup"][0] == ("box setup",)
    assert "setup_evidence.json" in vs.FETCH_NAMES
    assert "setup_evidence.json" in vs.PUBLISHED_NAMES
    assert "sf4-artifacts/setup_evidence.json" in WORKFLOW.read_text(encoding="utf-8")


def test_the_token_reaches_only_a_manual_setup_run():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert gst.HF_TOKEN_LINE in text and text.count("secrets.HF_TOKEN") == 1
    findings = gst.workflow_safety_findings(text)
    assert not [f for f in findings if "HF_TOKEN" in f]
    ungated = text.replace(gst.HF_TOKEN_LINE, "HF_TOKEN: ${{ secrets.HF_TOKEN }}")
    assert "HF_TOKEN must appear once, gated to the setup pipeline" in (
        gst.workflow_safety_findings(ungated)
    )


class _Remote(vs.SshRemote):
    def __init__(self):  # no key file, no network
        self.private = Path("/tmp/key")
        self.known_hosts = "/tmp/known_hosts"
        self.port = 29423
        self.host = "87.192.102.26"


def test_the_token_goes_on_stdin_never_in_the_argv(monkeypatch):
    seen = {}

    def fake_run(argv, **kwargs):
        seen.update(argv=argv, **kwargs)
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(vs.subprocess, "run", fake_run)
    assert _Remote()._send_hf_token(TOKEN) is True
    assert TOKEN not in " ".join(seen["argv"])
    assert seen["input"] == TOKEN.encode("utf-8")
    assert vs.REMOTE_HF_TOKEN in seen["argv"][-1] and "umask 077" in seen["argv"][-1]
    assert not any("TOKEN" in name for name in seen["env"])
    assert _Remote()._send_hf_token("") is False
    assert _Remote()._send_hf_token("has space") is False


def test_no_published_file_may_hold_the_token(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "setup_evidence.json").write_text(json.dumps({"leak": TOKEN}))
    monkeypatch.setenv(vs.HF_TOKEN_ENV, TOKEN)
    with pytest.raises(vs.SmokeError, match="No artifacts were published"):
        vs.publish_artifacts(run, tmp_path / "artifacts", api_key="")
    assert not (tmp_path / "artifacts").exists()


def test_the_fixes_found_on_the_box_stay_in_the_script():
    # Each one stopped or broke a stage on 60254439 (2026-11-11).
    text = SCRIPT.read_text(encoding="utf-8")
    assert '"setuptools>=69,<81"' in text  # torch 2.1.2 needs pkg_resources
    nerf = text.split("s_env_nerf() {", 1)[1].split("\n}", 1)[0]
    # nerfstudio upgrades setuptools, so the pin only sticks if it comes second.
    assert nerf.index("nerfstudio==1.1.4") < nerf.index('"setuptools>=69,<81"')
    assert '"transformers>=4.46,<5"' in text  # 5.x refuses torch 2.5.1
    assert "addict yapf timm" in text
    assert "from grounding_dino.groundingdino import _C" in text
    assert "from groundingdino import _C" not in text  # same .so twice aborts
    assert "pipdeptree==2.28.0" in text and "PIP_CONSTRAINT" in text
    da3 = text.split("s_da3() {", 1)[1].split("\n}", 1)[0]
    assert da3.index('install -e "$DA3_ROOT"') < da3.index("install numpy==2.4.6")
    # 4.x replaces torch 2.5.1, so the package pin follows the torch pin.
    sa = text.split("s_env_sa() {", 1)[1].split("\n}", 1)[0]
    assert sa.index("torch==2.5.1") < sa.index("open_clip_torch==3.3.0")
    assert "SAM2_CHECKPOINT=" in text
    assert "7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b" in text
    assert "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318" in text
    assert "f91db411c50efee93d8db7aeb323885650f6f722" in text
    assert "ISAACSIM_PIP=5.1.0" in text
    isaac = text.split("s_isaaclab() {", 1)[1].split("\n}", 1)[0]
    assert "PIP_CONFIG_FILE=/dev/null" in isaac
    assert isaac.index("setuptools>=69,<81") < isaac.index("./isaaclab.sh --install none")
    assert "flatdict==4.0.1" in isaac
    assert "ISAACLAB_TAG=v2.3.2" in text
    assert "37ddf626871758333d6ed89cf64ad702aef127d0" in text
    assert "OMNI_KIT_ACCEPT_EULA=YES" in text
    assert "--install none" in text
    assert "range(120)" in (ROOT / "sceneagent/isaaclab_headless_check.py").read_text(
        encoding="utf-8"
    )
    assert "cube_height_m" in text
    assert 'env["LIDRA_SKIP_INIT"] = "true"' in text
