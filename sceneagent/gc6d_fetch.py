# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fetch one GraspClutter6D scene without downloading the whole dataset.

The dataset (Back et al., RA-L, CC BY-SA 4.0) ships its scenes as one 7z split
into five volumes of about 203 GB. The archive is solid LZMA2 in 55 blocks of
about 3.75 GB, and each scene sits inside one block. Reading the archive
through HTTP Range requests therefore costs the index (about 13 MB) plus the
one block that holds the scene, instead of the whole dataset. Object models
come from ``models_eval.7z`` (about 87 MB), and only the objects the scene
uses are kept. ``manifest.json`` records each file's SHA-256, the source and
the license, and a second run with a matching manifest does nothing.

    python -m sceneagent.gc6d_fetch --scene 000091 --out /workspace/data/gc6d
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any, cast

SOURCE = "https://huggingface.co/datasets/GraspClutter6D/GraspClutter6D/resolve/main/"
SCENE_VOLUMES = [f"scenes.7z.{i:03d}" for i in range(1, 6)]
MODELS_ARCHIVE = "models_eval.7z"
LICENSE = "CC BY-SA 4.0"
ATTRIBUTION = (
    "GraspClutter6D: Back, Lee, Kim, Rho, Lee, Kang, Lee, Noh, Lee, Lee and Lee, "
    "IEEE Robotics and Automation Letters, arXiv:2504.06866"
)
# 64 MB requests run near the link rate; 1 MB requests ran at about a fifth.
BLOCK_BYTES = 64 << 20

Fetch = Callable[[int, int, int], bytes]


class RemoteVolumes(io.RawIOBase):
    """Read-only, seekable view of split volumes, fetched in blocks on demand."""

    def __init__(self, sizes: list[int], fetch: Fetch, block: int = BLOCK_BYTES):
        self.sizes = list(sizes)
        self.starts = [sum(self.sizes[:i]) for i in range(len(self.sizes))]
        self.total = sum(self.sizes)
        self.fetch = fetch
        self.block = block
        self.pos = 0
        self._cache: dict[int, bytes] = {}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.total}[whence]
        self.pos = base + offset
        return self.pos

    def _block(self, number: int) -> bytes:
        if number not in self._cache:
            low = number * self.block
            high = min(low + self.block, self.total) - 1
            parts = []
            for index, (start, size) in enumerate(zip(self.starts, self.sizes)):
                a, b = max(low, start), min(high, start + size - 1)
                if a <= b:
                    parts.append(self.fetch(index, a - start, b - start))
            if len(self._cache) >= 2:  # LZMA reads forward; two blocks is enough
                self._cache.pop(next(iter(self._cache)))
            self._cache[number] = b"".join(parts)
        return self._cache[number]

    def readinto(self, buffer: Any) -> int:
        wanted = min(len(buffer), max(self.total - self.pos, 0))
        got = 0
        while got < wanted:
            number, offset = divmod(self.pos, self.block)
            chunk = self._block(number)[offset : offset + wanted - got]
            buffer[got : got + len(chunk)] = chunk
            got += len(chunk)
            self.pos += len(chunk)
        return got


def http_fetch(urls: list[str]) -> Fetch:
    def fetch(index: int, start: int, end: int) -> bytes:
        request = urllib.request.Request(
            urls[index], headers={"Range": f"bytes={start}-{end}"}
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            data = response.read()
        if len(data) != end - start + 1:
            raise OSError(
                f"short read from {urls[index]}: {len(data)} of {end - start + 1}"
            )
        return data

    return fetch


def remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def extract_members(archive_file: Any, prefix: str, out_dir: Path | str) -> list[str]:
    """Extract the files under ``prefix`` from a 7z file object. Return their names."""
    import py7zr

    with py7zr.SevenZipFile(archive_file) as archive:
        names = [
            entry.filename
            for entry in archive.list()
            if entry.filename.startswith(prefix) and not entry.is_directory
        ]
        if not names:
            raise FileNotFoundError(f"nothing under {prefix} in the archive")
        archive.extract(path=out_dir, targets=names)
    return sorted(names)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    out_dir: Path | str, scene: str, names: list[str], source: str
) -> dict:
    out = Path(out_dir)
    manifest = {
        "scene": scene,
        "source": source,
        "license": LICENSE,
        "attribution": ATTRIBUTION,
        "files": {name: _sha256(out / name) for name in sorted(names)},
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _manifest_matches(out: Path, scene: str) -> bool:
    path = out / "manifest.json"
    if not path.is_file():
        return False
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return manifest.get("scene") == scene and all(
        (out / name).is_file() and _sha256(out / name) == digest
        for name, digest in manifest.get("files", {}).items()
    )


def fetch_scene(scene: str, out_dir: Path | str, source: str = SOURCE) -> dict:
    """Fetch ``scenes/<scene>/`` and its object models into ``out_dir``."""
    out = Path(out_dir)
    if _manifest_matches(out, scene):
        return json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    urls = [source + name for name in SCENE_VOLUMES]
    volumes = RemoteVolumes([remote_size(url) for url in urls], http_fetch(urls))
    names = extract_members(volumes, f"scenes/{scene}/", out)

    gt = json.loads((out / f"scenes/{scene}/scene_gt.json").read_text(encoding="utf-8"))
    obj_ids = sorted({entry["obj_id"] for image in gt.values() for entry in image})
    models_url = source + MODELS_ARCHIVE
    models = RemoteVolumes([remote_size(models_url)], http_fetch([models_url]))
    wanted = {f"obj_{obj_id:06d}.ply" for obj_id in obj_ids}
    import py7zr

    with py7zr.SevenZipFile(cast(IO[bytes], models)) as archive:
        model_names = [n for n in archive.getnames() if Path(n).name in wanted]
    missing = wanted - {Path(n).name for n in model_names}
    if missing:
        raise FileNotFoundError(f"models_eval.7z lacks {sorted(missing)}")
    models.seek(0)
    for name in model_names:
        names += extract_members(models, name, out)
        models.seek(0)
    return write_manifest(out, scene, names, source)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scene", default="000091")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = fetch_scene(args.scene, args.out)
    print(json.dumps({"scene": manifest["scene"], "files": len(manifest["files"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
