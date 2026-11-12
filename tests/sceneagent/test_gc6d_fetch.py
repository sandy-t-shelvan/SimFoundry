# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One GraspClutter6D scene out of a remote split 7z, without the network."""

from __future__ import annotations

import hashlib
import io
import json

import py7zr

from sceneagent.gc6d_fetch import RemoteVolumes, extract_members, write_manifest


def _volumes(data: bytes, parts: int) -> list[bytes]:
    step = -(-len(data) // parts)
    return [data[i : i + step] for i in range(0, len(data), step)]


def _fake_http(volumes, calls):
    def fetch(index, start, end):
        calls.append((index, start, end))
        return volumes[index][start : end + 1]

    return fetch


def _archive(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with py7zr.SevenZipFile(buffer, "w") as archive:  # solid LZMA2, like the dataset
        for name, payload in files.items():
            archive.writestr(payload, name)
    return buffer.getvalue()


def test_reads_match_the_joined_volumes_across_boundaries():
    data = bytes(range(256)) * 40
    volumes = _volumes(data, 3)
    calls: list = []
    remote = RemoteVolumes(
        [len(v) for v in volumes], _fake_http(volumes, calls), block=1000
    )
    for start, length in [(0, 10), (3400, 500), (len(data) - 7, 7), (1234, 5000)]:
        remote.seek(start)
        assert remote.read(length) == data[start : start + length]
    remote.seek(0, 2)
    assert remote.tell() == len(data)
    assert remote.read(5) == b""


def test_one_scene_is_extracted_from_a_split_solid_archive(tmp_path):
    files = {
        "scenes/000090/scene_gt.json": b'{"1": []}',
        "scenes/000091/scene_gt.json": b'{"1": [{"obj_id": 3}]}',
        "scenes/000091/rgb/000001.png": b"png-bytes" * 50,
        "scenes/000092/scene_gt.json": b'{"2": []}',
    }
    volumes = _volumes(_archive(files), 3)
    remote = RemoteVolumes([len(v) for v in volumes], _fake_http(volumes, []), block=64)
    written = extract_members(remote, "scenes/000091/", tmp_path)
    assert sorted(written) == [
        "scenes/000091/rgb/000001.png",
        "scenes/000091/scene_gt.json",
    ]
    assert (tmp_path / "scenes/000091/scene_gt.json").read_bytes() == files[
        "scenes/000091/scene_gt.json"
    ]
    assert not (tmp_path / "scenes/000090").exists()


def test_manifest_records_hashes_source_and_license(tmp_path):
    (tmp_path / "scenes/000091").mkdir(parents=True)
    (tmp_path / "scenes/000091/scene_gt.json").write_bytes(b"{}")
    manifest = write_manifest(
        tmp_path, "000091", ["scenes/000091/scene_gt.json"], "https://example/"
    )
    stored = json.loads((tmp_path / "manifest.json").read_text())
    assert stored == manifest
    assert (
        manifest["files"]["scenes/000091/scene_gt.json"]
        == hashlib.sha256(b"{}").hexdigest()
    )
    assert manifest["license"] == "CC BY-SA 4.0"
    assert "Back" in manifest["attribution"]
    assert manifest["source"] == "https://example/"
