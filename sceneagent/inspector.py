# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-22: the inspector reviews the exported scene and the model writes the fix report.

Paper §3.1 and Figure 7 (agent 14): an inspector reviews the exported result
in Isaac Lab and reports fixes. The work splits the way the SF-19 reviewer
does (:mod:`sceneagent.vlm_review`), so the Gemini key never reaches the box:

    # box (GPU workflow): views + settle summary
    python -m sceneagent.inspector render --scene scene.json --out OUT \\
        [--stage OUT/bin_physics.usda --isaac-python /path/to/isaac/python]

    # GitHub runner (holds GEMINI_API_KEY): the report
    python -m sceneagent.inspector report --views inspection_views.zip \\
        --settle settle_summary.json --out inspection/isaac_lab.json --source-run ID

``render`` draws a fixed set of views of the placed scene and, when Isaac Lab
is on the host, steps the exported stage for two seconds and writes what it
measured. Without Isaac Lab the views are drawn on the CPU from the meshes
(points sampled from each mesh, depth sorted), and the settle summary says
plainly that no simulation ran; it never invents a settle result.

``report`` sends the contact sheet and the settle summary to
``gemini-3.8-flash`` with a fixed prompt (ours: the paper does not publish
its inspector prompt) and a response schema, and writes
``inspection/isaac_lab.json``. ``fix_report_from_model`` is true only when
the model replied, the reply parsed, and it listed at least one finding. A
failed call, an unparsable reply or an empty finding list leaves it false,
which is the gate `skills.STAGE_OPENED_SKILLS` describes: an opened stage is
not a fix report.

The scene manifest (``scene.json``) is ours. It names, per object, the mesh
file and :mod:`sceneagent.bin_place`'s similarity into the reconstruction
frame (``x' = s R x + t``), the same inputs ``physics_export.PlacedObject``
takes, plus optional ``body_type`` and ``metres_per_unit``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from sceneagent.similarity_icp import apply_similarity

TILE_PX = 384
RENDER_POINTS = 20000
# Fixed views of the whole scene, Z up. Ours: enough to see an object sunk in
# the ground, floating, interpenetrating a neighbour, or left out.
VIEWS: dict[str, tuple[float, float, float]] = {
    "top": (0.0, 0.0, 1.0),
    "front": (0.0, -1.0, 0.0),
    "side": (1.0, 0.0, 0.0),
    "oblique_1": (1.0, -1.0, 1.0),
    "oblique_2": (-1.0, 1.0, 1.0),
}
VIEWS_ZIP = "inspection_views.zip"
SHEET_NAME = "inspection_sheet.png"
SETTLE_NAME = "settle_summary.json"
# SF-35 dc2's settle bar: after 2 s every actor is slower than 1 cm/s and none
# is below the ground.
SETTLE_SECONDS = 2.0
SETTLE_MAX_SPEED_M_S = 0.01
FINDING_CHARS = 300
SUMMARY_CHARS = 600
MAX_FINDINGS = 20
ISSUES = (
    "below_ground",
    "floating",
    "interpenetrating",
    "not_settled",
    "wrong_scale",
    "wrong_pose",
    "missing_object",
    "other",
)

Runner = Callable[..., Any]


# ---------------------------------------------------------------- box side


def load_scene(path: Path) -> list[dict[str, Any]]:
    """The scene manifest's objects, each with its mesh resolved and checked."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    objects = payload.get("objects") if isinstance(payload, dict) else None
    if not isinstance(objects, list) or not objects:
        raise ValueError("scene manifest has no objects")
    base = Path(path).parent
    resolved = []
    for item in objects:
        if not isinstance(item, dict) or not item.get("name") or not item.get("mesh"):
            raise ValueError("every scene object needs a name and a mesh")
        mesh = Path(item["mesh"])
        if not mesh.is_absolute():
            mesh = base / mesh
        resolved.append({**item, "mesh": mesh})
    return resolved


def _object_points(item: Mapping[str, Any], count: int) -> np.ndarray:
    import trimesh

    mesh = trimesh.load(item["mesh"], force="mesh", process=False)
    samples, _faces = trimesh.sample.sample_surface(mesh, count, seed=7)
    rotation = np.asarray(item.get("rotation", np.eye(3)), dtype=float)
    translation = np.asarray(item.get("translation", np.zeros(3)), dtype=float)
    points = apply_similarity(
        np.asarray(samples, dtype=float),
        float(item.get("scale", 1.0)),
        rotation,
        translation,
    )
    return points * float(item.get("metres_per_unit", 1.0))


def _basis(direction: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    reference = (
        np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    )
    right = np.cross(reference, d)
    right /= np.linalg.norm(right)
    up = np.cross(d, right)
    return d, right, up


def _colour(index: int) -> tuple[int, int, int]:
    """A distinct, stable colour per object (golden-ratio hue steps)."""
    import colorsys

    r, g, b = colorsys.hsv_to_rgb((index * 0.618034) % 1.0, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def render_views(
    clouds: Mapping[str, np.ndarray], size: int = TILE_PX
) -> dict[str, np.ndarray]:
    """Orthographic views of the placed scene from the CPU, with the ground drawn.

    Every view shares one frame around all objects, so relative heights and
    gaps read the same from each side. The ground (z at the lowest object
    point, or 0 when every object is above it) is a grey line or plane.
    """
    names = sorted(clouds)
    every = np.vstack([clouds[n] for n in names])
    centre = every.mean(axis=0)
    ground_z = min(0.0, float(every[:, 2].min()))
    half = 0.6 * float(np.ptp(every, axis=0).max() or 1.0)
    views: dict[str, np.ndarray] = {}
    for view_name, direction in VIEWS.items():
        d, right, up = _basis(direction)
        image = np.full((size, size, 3), 255, dtype=np.uint8)
        # Ground: sample a grid on z = ground_z across the frame.
        grid = np.linspace(-half, half, 60)
        gx, gy = np.meshgrid(grid + centre[0], grid + centre[1])
        ground = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, ground_z)])
        layers = [(ground, (205, 205, 205))] + [
            (clouds[n], _colour(i)) for i, n in enumerate(names)
        ]
        cloud = np.vstack([pts for pts, _ in layers])
        colours = np.vstack(
            [
                np.tile(np.array(col, dtype=np.uint8), (len(pts), 1))
                for pts, col in layers
            ]
        )
        x = (cloud - centre) @ right / (2 * half) + 0.5
        y = 0.5 - (cloud - centre) @ up / (2 * half)
        inside = (x >= 0) & (x < 1) & (y >= 0) & (y < 1)
        depth = (cloud - centre) @ d
        order = np.argsort(depth[inside])  # far first; the nearest is drawn last
        cols = (x[inside] * size).astype(int)[order]
        rows = (y[inside] * size).astype(int)[order]
        image[rows, cols] = colours[inside][order]
        views[view_name] = image
    return views


def save_views(
    views: Mapping[str, np.ndarray], names: Sequence[str], out: Path
) -> tuple[Path, Path]:
    """Write each view, a labelled contact sheet with a legend, and the zip."""
    from PIL import Image, ImageDraw

    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for view_name in VIEWS:
        tile = Image.fromarray(views[view_name])
        ImageDraw.Draw(tile).text((4, 4), view_name, fill=(0, 0, 0))
        tile.save(out / f"view_{view_name}.png")
        tiles.append(tile)
    legend = Image.new("RGB", (TILE_PX, TILE_PX), "white")
    draw = ImageDraw.Draw(legend)
    draw.text((4, 4), "legend", fill=(0, 0, 0))
    for i, name in enumerate(sorted(names)[:28]):
        draw.rectangle([6, 22 + 12 * i, 14, 30 + 12 * i], fill=_colour(i))
        draw.text((20, 20 + 12 * i), str(name)[:40], fill=(0, 0, 0))
    tiles.append(legend)
    columns = 3
    rows = -(-len(tiles) // columns)
    sheet = Image.new("RGB", (columns * TILE_PX, rows * TILE_PX), "white")
    for k, tile in enumerate(tiles):
        sheet.paste(tile, ((k % columns) * TILE_PX, (k // columns) * TILE_PX))
    sheet_path = out / SHEET_NAME
    sheet.save(sheet_path)
    zip_path = out / VIEWS_ZIP
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(sheet_path, SHEET_NAME)
        for view_name in VIEWS:
            archive.write(out / f"view_{view_name}.png", f"view_{view_name}.png")
    return sheet_path, zip_path


# Run inside Isaac Lab's Python. Not yet run on a box (needs 60254439's setup);
# its output shape is what parse_isaac_settle reads, and the tests check that
# plumbing with a fake runner, not Isaac itself.
ISAAC_SETTLE_SCRIPT = r"""
import json, sys
stage_path, seconds, out_path = sys.argv[1], float(sys.argv[2]), sys.argv[3]
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import omni.usd
from pxr import UsdPhysics, UsdGeom
from isaacsim.core.api import World
omni.usd.get_context().open_stage(stage_path)
world = World(stage_units_in_meters=1.0)
world.reset()
stage = omni.usd.get_context().get_stage()
from isaacsim.core.prims import RigidPrim
bodies = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
view = RigidPrim(prim_paths_expr=bodies) if bodies else None
steps = int(seconds * 60)
for _ in range(steps):
    world.step(render=False)
result = {"simulator": "isaac_lab", "seconds": seconds, "bodies": []}
if view is not None:
    positions, _ = view.get_world_poses()
    velocities = view.get_linear_velocities()
    for path, p, v in zip(bodies, positions.tolist(), velocities.tolist()):
        result["bodies"].append({"path": path, "z": p[2], "speed": sum(c * c for c in v) ** 0.5})
json.dump(result, open(out_path, "w"))
app.close()
"""


def parse_isaac_settle(
    payload: Mapping[str, Any], ground_z: float = 0.0
) -> dict[str, Any]:
    """Judge an Isaac settle result against SF-35 dc2's bar."""
    bodies = payload.get("bodies")
    if not isinstance(bodies, list):
        raise ValueError("settle result has no body list")
    rows = []
    for body in bodies:
        speed = float(body["speed"])
        z = float(body["z"])
        rows.append(
            {
                "path": str(body["path"]),
                "speed_m_s": speed,
                "z_m": z,
                "at_rest": speed < SETTLE_MAX_SPEED_M_S,
                "below_ground": z < ground_z,
            }
        )
    return {
        "simulator": "isaac_lab",
        "ran": True,
        "seconds": float(payload.get("seconds", SETTLE_SECONDS)),
        "rule": "ours from SF-35 dc2: every body slower than 1 cm/s after 2 s, none below the ground",
        "bodies": rows,
        "settled": bool(rows)
        and all(r["at_rest"] and not r["below_ground"] for r in rows),
    }


def settle(
    stage: Path | None,
    isaac_python: str | None,
    out: Path,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Step the stage in Isaac Lab when it is on the host; otherwise say so."""
    not_run = {
        "simulator": None,
        "ran": False,
        "settled": None,
        "reason": "",
    }
    if stage is None or not Path(stage).is_file():
        return {**not_run, "reason": "no exported stage to simulate"}
    if not isaac_python or not Path(isaac_python).is_file():
        return {
            **not_run,
            "reason": "Isaac Lab is not available on this host; no settle run",
        }
    script = out / "isaac_settle.py"
    script.write_text(ISAAC_SETTLE_SCRIPT, encoding="utf-8")
    result_path = out / "isaac_settle_raw.json"
    argv = [
        isaac_python,
        str(script),
        str(stage),
        str(SETTLE_SECONDS),
        str(result_path),
    ]
    completed = runner(argv, capture_output=True, text=True, timeout=1800, check=False)
    if completed.returncode != 0 or not result_path.is_file():
        tail = (completed.stderr or completed.stdout or "")[-400:]
        return {
            **not_run,
            "simulator": "isaac_lab",
            "reason": f"Isaac settle failed (exit {completed.returncode}): {tail}",
            "command": " ".join(shlex.quote(a) for a in argv),
        }
    return parse_isaac_settle(json.loads(result_path.read_text(encoding="utf-8")))


def render_inspection(
    scene: Path,
    out: Path,
    stage: Path | None = None,
    isaac_python: str | None = None,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Box side: views, contact sheet, zip and settle summary in ``out``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    objects = load_scene(scene)
    clouds = {item["name"]: _object_points(item, RENDER_POINTS) for item in objects}
    views = render_views(clouds)
    sheet, archive = save_views(views, list(clouds), out)
    summary = settle(stage, isaac_python, out, runner)
    summary.update(
        objects=sorted(clouds),
        views=list(VIEWS),
        renderer="cpu point render (ours)",
    )
    (out / SETTLE_NAME).write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return {"sheet": str(sheet), "views_zip": str(archive), "settle": summary}


# ---------------------------------------------------------------- runner side

INSPECT_PROMPT = """\
You are the inspector for a scene rebuilt from photos of household objects and
exported for physics simulation. The image is a contact sheet of the exported
scene: top, front, side and two oblique views in one shared frame, Z up, each
object in its own colour (the legend tile names them), and the ground in grey.
Below is the settle summary: whether a physics run happened and, if so, each
body's speed and height after two seconds.

Report what should be fixed before this scene is used. For each problem name
the object (as in the legend, or "scene"), the issue type, and a concrete fix
(for example "raise by 2 cm so it rests on the ground"). Look for objects
below or sunk into the ground, floating above any support, interpenetrating a
neighbour, implausibly sized, posed impossibly, or not at rest in the settle
summary. If nothing needs fixing, return an empty list and say so in the
summary.

Answer as JSON: {"findings": [{"object": "<name>", "issue": "<issue type>",
"fix": "<one sentence>"}], "summary": "<two sentences on the scene>"}.
"""

REPORT_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "findings": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "object": {"type": "STRING"},
                    "issue": {"type": "STRING", "enum": list(ISSUES)},
                    "fix": {"type": "STRING"},
                },
                "required": ["object", "issue", "fix"],
            },
        },
        "summary": {"type": "STRING"},
    },
    "required": ["findings", "summary"],
}


class ImageHarness(Protocol):
    """The part of :class:`sceneagent.harness.GeminiHarness` used here."""

    model_id: str
    call_count: int

    def complete(
        self,
        prompt: str,
        images: Any = (),
        response_schema: dict[str, Any] | None = None,
    ) -> str: ...


def prompt_sha256() -> str:
    """Hash of the inspector prompt, recorded in every report."""
    return hashlib.sha256(INSPECT_PROMPT.encode("utf-8")).hexdigest()


def parse_report(text: str) -> dict[str, Any]:
    """The model's reply as ``{findings, summary}``; raise on anything else."""
    body = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", body, flags=re.DOTALL)
    if fenced:
        body = fenced.group(1)
    payload = json.loads(body)
    if not isinstance(payload, dict) or not isinstance(payload.get("findings"), list):
        raise ValueError(f"inspector reply {body[:80]!r} has no findings list")
    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("inspector reply has no summary")
    findings = []
    for item in payload["findings"][:MAX_FINDINGS]:
        if not isinstance(item, dict):
            raise ValueError("a finding is not an object")
        issue = item.get("issue")
        if issue not in ISSUES:
            raise ValueError(f"finding issue {issue!r} is not one of {ISSUES}")
        obj, fix = item.get("object"), item.get("fix")
        if not isinstance(obj, str) or not obj.strip():
            raise ValueError("a finding names no object")
        if not isinstance(fix, str) or not fix.strip():
            raise ValueError("a finding proposes no fix")
        findings.append(
            {
                "object": obj.strip()[:120],
                "issue": issue,
                "fix": fix.strip()[:FINDING_CHARS],
            }
        )
    return {"findings": findings, "summary": summary.strip()[:SUMMARY_CHARS]}


def read_views(zip_path: Path) -> bytes:
    """The contact sheet from ``inspection_views.zip``."""
    with zipfile.ZipFile(zip_path) as archive:
        if SHEET_NAME not in archive.namelist():
            raise ValueError(f"{zip_path} has no {SHEET_NAME}")
        return archive.read(SHEET_NAME)


def inspect(
    sheet_png: bytes,
    settle_summary: Mapping[str, Any],
    harness: ImageHarness,
    source_run: str = "",
) -> dict[str, Any]:
    """One call: the model reads the sheet and the settle summary and lists fixes.

    ``fix_report_from_model`` is true only when the call succeeded, the reply
    parsed, and it named at least one finding. Every other outcome is recorded
    with the field false; nothing is guessed in the model's place.
    """
    document: dict[str, Any] = {
        "inspector": "ours: sceneagent.inspector report",
        "source_run": source_run,
        "model": harness.model_id,
        "prompt_sha256": prompt_sha256(),
        "prompt_source": "ours",
        "settle": dict(settle_summary),
        "model_replied": False,
        "findings": [],
        "summary": None,
        "fix_report_from_model": False,
    }
    prompt = (
        INSPECT_PROMPT
        + "\nSettle summary:\n"
        + json.dumps(settle_summary, indent=2, sort_keys=True)[:4000]
    )
    try:
        reply = harness.complete(
            prompt, images=[(sheet_png, "image/png")], response_schema=REPORT_SCHEMA
        )
        document["model_replied"] = True
        report = parse_report(reply)
        document.update(report)
        document["fix_report_from_model"] = bool(report["findings"])
    except Exception as exc:  # recorded; the key is already redacted by the harness
        document["error"] = f"{type(exc).__name__}: {exc}"[:300]
    document["vlm_calls"] = harness.call_count
    return document


def main(argv: list[str] | None = None, harness: ImageHarness | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    render = sub.add_parser("render", help="box: draw views and run the settle check")
    render.add_argument("--scene", type=Path, required=True)
    render.add_argument("--out", type=Path, required=True)
    render.add_argument("--stage", type=Path, default=None)
    render.add_argument("--isaac-python", default=None)
    report = sub.add_parser("report", help="runner: the model writes the fix report")
    report.add_argument("--views", type=Path, required=True)
    report.add_argument("--settle", type=Path, required=True)
    report.add_argument("--out", type=Path, required=True)
    report.add_argument("--source-run", default="")
    args = parser.parse_args(argv)
    if args.command == "render":
        result = render_inspection(args.scene, args.out, args.stage, args.isaac_python)
        print(
            json.dumps(
                {
                    "views_zip": result["views_zip"],
                    "settled": result["settle"]["settled"],
                }
            )
        )
        return 0
    if args.source_run and not re.fullmatch(r"[0-9]{1,20}", args.source_run):
        parser.error("--source-run must be a run id (digits only)")
    sheet = read_views(args.views)
    settle_summary = json.loads(args.settle.read_text(encoding="utf-8"))
    if harness is None:
        from sceneagent.harness import selected_harness

        harness = selected_harness()
    document = inspect(sheet, settle_summary, harness, args.source_run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "model": document["model"],
                "vlm_calls": document["vlm_calls"],
                "findings": len(document["findings"]),
                "fix_report_from_model": document["fix_report_from_model"],
            }
        )
    )
    return 0 if document["model_replied"] and "error" not in document else 1


if __name__ == "__main__":
    sys.exit(main())
