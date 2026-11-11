# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-19 round 2: Gemini accepts or rejects each retry from its before and after views.

Round 1 let Gemini choose the retry but kept the old acceptance: a flip
was kept when the fit distance fell, a regeneration when the silhouette
rose. Neither tracks ground truth (the bottle's silhouette rose while its
F1 fell). Here the VLM also accepts. SAM 3D regeneration is not repeatable,
so each retry is run once and its result saved; every later stage judges
and applies exactly that result.

    python -m sceneagent.bin_review_pairs build --out OUT --verdicts V
    python -m sceneagent.bin_review_pairs apply --out OUT --accept A

``build`` (gpu-smoke ``review_candidates``) checks the retry verdicts were
judged on this placement, runs the chosen retries, saves each result under
``reviewer/pairs/<hash>/`` (the hash covers every saved cloud), and draws
each changed object's before and after sheets as a blind pair: ``A`` and
``B`` in an order drawn from a recorded seed, so a position preference in
the model cannot pass for a judgement. A retry that moved nothing is not
drawn. ``candidates.json`` maps A and B back; it stays with the box's
evidence and is not sent to the model.

``apply`` (gpu-smoke ``review_accept``) loads the saved results by hash,
refuses on any mismatch (hash, saved clouds, placement), keeps the side
the VLM judged better, and scores. It also scores what the old test would
have kept and, for evaluation only, an oracle: whether keeping each retry
on its own raises the scene's F1 against ground truth. The oracle is
computed here, after the VLM has judged, and nowhere earlier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from sceneagent import bin_place
from sceneagent import bin_review as br
from sceneagent.bin_run import DATA_DIR, SCENE, SOURCE

CANDIDATES_ITEM = "review candidates"
PAIRS_ITEM = "review pairs drawn"
ACCEPT_ITEM = "reviewer retries (VLM accepted)"
ACCEPT_SCORE_ITEM = "bin score (VLM accepted)"
BUILD_REQUIRED = (CANDIDATES_ITEM, PAIRS_ITEM)
APPLY_REQUIRED = (ACCEPT_ITEM, ACCEPT_SCORE_ITEM)
PAIRS_DIR = Path("reviewer") / "pairs"
PAIR_SEED = 20261111
SIDES = ("A", "B")
HASH_RE = re.compile(r"[0-9a-f]{64}")

Runner = Callable[..., Any]


def cloud_sha256(cloud: np.ndarray) -> str:
    """Hash of a cloud's float64 bytes, independent of how it is saved."""
    return hashlib.sha256(
        np.ascontiguousarray(cloud, dtype=np.float64).tobytes()
    ).hexdigest()


def candidates_hash(entries: list[dict[str, Any]]) -> str:
    """One hash for the candidate set: each object's kind and cloud hash."""
    key = sorted((e["object"], e["kind"], e["cloud_sha256"]) for e in entries)
    return hashlib.sha256(json.dumps(key).encode("utf-8")).hexdigest()


def pair_order(numbers: list[int], seed: int = PAIR_SEED) -> dict[int, dict[str, str]]:
    """Which of before and after is shown as A, per object, from a recorded seed."""
    rng = random.Random(seed)
    order = {}
    for n in sorted(numbers):
        if rng.random() < 0.5:
            order[n] = {"A": "before", "B": "after"}
        else:
            order[n] = {"A": "after", "B": "before"}
    return order


def assembled_views(inputs: dict[str, Any], number: int) -> list[str]:
    """The object's views, largest mask first, as run_bin_review orders them."""
    area = {
        (view, d["mask_index"]): int(d.get("mask_area_px", 0))
        for view, dets in inputs["detections"].items()
        for d in dets
        if d.get("kept")
    }
    obj = next(o for o in inputs["association"]["object_list"] if o["object"] == number)
    by_area = sorted(
        obj["masks"], key=lambda m: -area.get((m["view"], m["mask_index"]), 0)
    )
    return list(dict.fromkeys(m["view"] for m in by_area))


def _load_ledger(out: Path) -> dict[str, Any]:
    try:
        return json.loads((out / "ledger.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"entries": []}


def _write_ledger(
    out: Path, ledger: dict[str, Any], entries: list[dict[str, Any]]
) -> None:
    ledger.update(scene_name=f"{SOURCE} {SCENE}", source=SOURCE, entries=entries)
    (out / "ledger.json").write_text(
        json.dumps(ledger, indent=2) + "\n", encoding="utf-8"
    )


def _placement(
    out: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[int, dict[str, Any]]]:
    inputs = bin_place._load(out)
    placement = bin_place.place_objects(out, inputs)
    clouds: dict[str, np.ndarray] = dict(placement["placed"])
    records = {r["object"]: r for r in placement["objects"] if r.get("placed")}
    return inputs, clouds, records


def build_candidates(
    out: Path | str,
    verdicts: Path | str,
    *,
    runner: Runner = subprocess.run,
    sam3d_config: Callable[[], Path] | None = None,
    seed: int = PAIR_SEED,
) -> int:
    """Run the VLM's retries once, save the results, draw blind before/after pairs."""
    out = Path(out)
    ledger = _load_ledger(out)
    entries = [
        e for e in ledger.get("entries", []) if e.get("item") not in BUILD_REQUIRED
    ]
    cand_entry: dict[str, Any] = {"item": CANDIDATES_ITEM}
    pairs_entry: dict[str, Any] = {"item": PAIRS_ITEM}
    try:
        verdict_bytes = Path(verdicts).read_bytes()
        meta, vlm = br.load_verdicts(Path(verdicts))
        inputs, clouds, records = _placement(out)
        br.check_placement(meta["judged_placement"], records)
        used = {n: r["view"] for n, r in records.items()}
        following = br.next_views(inputs["association"], inputs["detections"], used)
        objects: list[dict[str, Any]] = []
        candidate_clouds: dict[int, np.ndarray] = {}
        regen: list[int] = []
        for number in sorted(records):
            decision, source = br.vlm_decision(vlm.get(number), number in following)
            item: dict[str, Any] = {
                "object": number,
                "decision": decision,
                "decision_source": source,
                "fit_total": records[number]["fit_total"],
            }
            if decision == "icp_flip":
                cloud, outcome = br.icp_flip_candidate(out, inputs, records[number])
                item.update(outcome)
                candidate_clouds[number] = cloud
            elif decision == "sam3d_next_view":
                regen.append(number)
            objects.append(item)
        if regen:
            regenerated = br.regenerate(
                out,
                inputs,
                regen,
                following,
                out / "sam3d" / "retry_pairs",
                runner,
                sam3d_config,
            )
            for item in objects:
                n = item["object"]
                if n not in regenerated:
                    continue
                regen_cloud, new, result = regenerated[n]
                if regen_cloud is None or new is None:
                    item.update(
                        kind="sam3d_next_view", failed=result.get("error", "no result")
                    )
                    continue
                before = records[n]["residuals"]["silhouette"] or 0.0
                after = new["residuals"]["silhouette"] or 0.0
                item.update(
                    kind="sam3d_next_view",
                    view=following[n]["view"],
                    glb=new["glb"],
                    similarity=new["similarity"],
                    before_silhouette=before,
                    after_silhouette=after,
                    test="silhouette rises",
                    test_accepts=bool(after > before),
                    unchanged=False,
                    gate=new["gate"],
                )
                candidate_clouds[n] = regen_cloud
        for item in objects:
            if item["object"] in candidate_clouds:
                item["cloud_sha256"] = cloud_sha256(candidate_clouds[item["object"]])
        with_cloud = [i for i in objects if "cloud_sha256" in i]
        digest = candidates_hash(with_cloud)
        folder = out / PAIRS_DIR / digest[:16]
        folder.mkdir(parents=True, exist_ok=True)
        for item in with_cloud:
            n = item["object"]
            np.save(folder / f"object_{n}.npy", candidate_clouds[n])
            glb = folder / f"object_{n}.glb"
            if Path(item["glb"]) != glb:
                shutil.copyfile(item["glb"], glb)
            item["glb"] = str(glb)
        cand_entry.update(
            decision="accept",
            use_gpu=bool(regen),
            candidates=len(with_cloud),
            failed=sum(1 for i in objects if "failed" in i),
            candidates_hash=digest,
        )
        changed = [i["object"] for i in with_cloud if not i.get("unchanged")]
        order = pair_order(changed, seed)
        drawn = {}
        for item in with_cloud:
            n = item["object"]
            if item.get("unchanged"):
                continue
            name = f"object_{n}"
            views = assembled_views(inputs, n)
            before = br.review_object(
                out, inputs, records[n], clouds, views, folder=folder / "before" / name
            )
            after_record = {
                **records[n],
                "similarity": item["similarity"],
                "glb": item["glb"],
                "view": item["view"],
            }
            after = br.review_object(
                out,
                inputs,
                after_record,
                {**clouds, name: candidate_clouds[n]},
                views,
                folder=folder / "after" / name,
            )
            item.update(order[n])
            drawn[n] = {"before": before["sheet"], "after": after["sheet"]}
        with zipfile.ZipFile(
            out / "review_pairs.zip", "w", zipfile.ZIP_DEFLATED
        ) as archive:
            for n, sheets in drawn.items():
                for side in SIDES:
                    archive.write(sheets[order[n][side]], f"object_{n}_{side}.png")
        pairs_entry.update(decision="accept", use_gpu=False, pairs=len(drawn))
        manifest = {
            "scene": SCENE,
            "candidates_hash": digest,
            "folder": str(folder),
            "verdicts_sha256": hashlib.sha256(verdict_bytes).hexdigest(),
            "source_run": meta.get("source_run"),
            "judged_placement": [
                {"object": n, "fit_total": r["fit_total"]}
                for n, r in sorted(records.items())
            ],
            "pair_seed": seed,
            "objects": objects,
        }
        text = json.dumps(manifest, indent=2) + "\n"
        (folder / "candidates.json").write_text(text, encoding="utf-8")
        (out / "candidates.json").write_text(text, encoding="utf-8")
    except (OSError, ValueError, KeyError, RuntimeError, StopIteration) as exc:
        reason = f"{type(exc).__name__}: {exc}"[-800:]
        for entry in (cand_entry, pairs_entry):
            if "decision" not in entry:
                entry.update(decision="reject", reason=reason)
    entries += [cand_entry, pairs_entry]
    _write_ledger(out, ledger, entries)
    outcomes = {e.get("item"): e.get("decision") for e in entries}
    return 0 if all(outcomes.get(i) == "accept" for i in BUILD_REQUIRED) else 1


def load_accept_verdicts(
    path: Path,
) -> tuple[dict[str, Any], dict[int, dict[str, str]]]:
    """The pair verdicts from ``vlm_review --mode accept``, every field checked."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("mode") != "accept":
        raise ValueError("not an accept verdict file")
    digest = payload.get("candidates_hash")
    if not isinstance(digest, str) or not HASH_RE.fullmatch(digest):
        raise ValueError(f"candidates_hash {digest!r} is not a sha256")
    verdicts: dict[int, dict[str, str]] = {}
    for item in payload.get("objects") or []:
        number = item.get("object") if isinstance(item, dict) else None
        better = item.get("better") if isinstance(item, dict) else None
        if not isinstance(number, int) or isinstance(number, bool):
            raise ValueError(f"pair object {number!r} is not an integer")
        if better not in SIDES:
            raise ValueError(f"pair verdict {better!r} is not A or B")
        reason = item.get("reason")
        verdicts[number] = {
            "better": better,
            "reason": reason[:400] if isinstance(reason, str) else "",
        }
    meta = {
        key: payload.get(key)
        for key in (
            "candidates_hash",
            "source_run",
            "model",
            "prompt_sha256",
            "vlm_calls",
        )
    }
    return meta, verdicts


def load_candidates(
    out: Path, digest: str
) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    """The saved candidate set named by ``digest``; refuse if anything differs."""
    folder = out / PAIRS_DIR / digest[:16]
    manifest = json.loads((folder / "candidates.json").read_text(encoding="utf-8"))
    if manifest.get("candidates_hash") != digest:
        raise ValueError("the saved candidates are not the set the VLM judged")
    clouds: dict[int, np.ndarray] = {}
    for item in manifest["objects"]:
        if "cloud_sha256" not in item:
            continue
        cloud = np.load(folder / f"object_{item['object']}.npy")
        if cloud_sha256(cloud) != item["cloud_sha256"]:
            raise ValueError(f"saved cloud for object {item['object']} has changed")
        clouds[item["object"]] = cloud
    if (
        candidates_hash([i for i in manifest["objects"] if "cloud_sha256" in i])
        != digest
    ):
        raise ValueError("the candidate manifest does not hash to its name")
    return manifest, clouds


def choose(item: dict[str, Any], verdict: dict[str, str] | None) -> tuple[str, str]:
    """Which side the VLM keeps for one candidate, and why."""
    if item.get("unchanged"):
        return "before", "retry moved nothing; not judged"
    if verdict is None:
        return "before", "no VLM verdict"
    return item[verdict["better"]], "VLM"


def apply_accepted(
    out: Path | str, accept: Path | str, data: Path | str = DATA_DIR
) -> int:
    """Keep the side the VLM judged better; score it against the old test and the oracle."""
    out, data = Path(out), Path(data)
    ledger = _load_ledger(out)
    entries = [
        e for e in ledger.get("entries", []) if e.get("item") not in APPLY_REQUIRED
    ]
    accept_entry: dict[str, Any] = {"item": ACCEPT_ITEM}
    score_entry: dict[str, Any] = {"item": ACCEPT_SCORE_ITEM}
    evidence: dict[str, Any] = {"scene": SCENE, "objects": []}
    try:
        meta, verdicts = load_accept_verdicts(Path(accept))
        manifest, saved = load_candidates(out, meta["candidates_hash"])
        inputs, clouds, records = _placement(out)
        judged = {i["object"]: i["fit_total"] for i in manifest["judged_placement"]}
        br.check_placement(judged, records)
        evidence["vlm"] = meta
        base = bin_place.score_scene(data, inputs, clouds)
        vlm_scene, test_scene, oracle_scene = dict(clouds), dict(clouds), dict(clouds)
        agree = {"vlm": 0, "test": 0, "judged": 0}
        for item in manifest["objects"]:
            n = item["object"]
            row: dict[str, Any] = {
                "object": n,
                "decision": item["decision"],
                "kind": item.get("kind"),
            }
            if n not in saved:
                row["candidate"] = item.get("failed", "none")
                evidence["objects"].append(row)
                continue
            name = f"object_{n}"
            vlm_side, why = choose(item, verdicts.get(n))
            test_side = "after" if item["test_accepts"] else "before"
            # Evaluation only: does keeping this retry alone raise the scene's F1?
            alone = bin_place.score_scene(data, inputs, {**clouds, name: saved[n]})
            delta = alone["f1_mean"] - base["f1_mean"]
            oracle = "after" if delta > 0 else "before" if delta < 0 else "tie"
            row.update(
                vlm=vlm_side,
                vlm_reason=why if why != "VLM" else verdicts[n]["reason"],
                vlm_source=why,
                test=test_side,
                oracle=oracle,
                f1_delta_if_kept=delta,
            )
            for side, scene in ((vlm_side, vlm_scene), (test_side, test_scene)):
                if side == "after":
                    scene[name] = saved[n]
            if oracle == "after":
                oracle_scene[name] = saved[n]
            if oracle != "tie" and not item.get("unchanged"):
                agree["judged"] += 1
                agree["vlm"] += vlm_side == oracle
                agree["test"] += test_side == oracle
            evidence["objects"].append(row)
        scores = {
            "no_review": base,
            "test_accepted": bin_place.score_scene(data, inputs, test_scene),
            "vlm_accepted": bin_place.score_scene(
                data, inputs, vlm_scene, tag="_vlm_accepted"
            ),
            "oracle_upper_bound": bin_place.score_scene(data, inputs, oracle_scene),
        }
        evidence["scores"] = {
            k: {m: v[m] for m in ("f1_mean", "chamfer_mean_matched_m")}
            for k, v in scores.items()
        }
        evidence["agreement_with_oracle"] = agree
        evidence["oracle_rule"] = (
            "ours, evaluation only: a retry is better when keeping it alone raises "
            "the scene F1@1cm; computed after the VLM judged"
        )
        accept_entry.update(
            decision="accept",
            use_gpu=False,
            candidates=len(saved),
            vlm_keeps_after=sum(
                1 for r in evidence["objects"] if r.get("vlm") == "after"
            ),
            vlm_agrees_with_oracle=agree["vlm"],
            test_agrees_with_oracle=agree["test"],
            judged_against_oracle=agree["judged"],
        )
        vlm_score = scores["vlm_accepted"]
        (out / vlm_score["file_name"]).write_text(
            json.dumps(vlm_score, indent=2) + "\n", encoding="utf-8"
        )
        score_entry.update(
            decision="accept",
            use_gpu=False,
            **{k: round(v["f1_mean"], 4) for k, v in scores.items()},
        )
    except (OSError, ValueError, KeyError, RuntimeError, StopIteration) as exc:
        reason = f"{type(exc).__name__}: {exc}"[-800:]
        for entry in (accept_entry, score_entry):
            if "decision" not in entry:
                entry.update(decision="reject", reason=reason)
    entries += [accept_entry, score_entry]
    _write_ledger(out, ledger, entries)
    (out / "bin_review_accept.json").write_text(
        json.dumps(evidence, indent=2, default=str) + "\n", encoding="utf-8"
    )
    outcomes = {e.get("item"): e.get("decision") for e in entries}
    return 0 if all(outcomes.get(i) == "accept" for i in APPLY_REQUIRED) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    build = sub.add_parser("build", help="run the retries once and draw blind pairs")
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--verdicts", type=Path, required=True)
    apply = sub.add_parser("apply", help="keep what the VLM judged better and score it")
    apply.add_argument("--out", type=Path, required=True)
    apply.add_argument("--accept", type=Path, required=True)
    apply.add_argument("--data", type=Path, default=Path(DATA_DIR))
    args = parser.parse_args(argv)
    if args.stage == "build":
        return build_candidates(args.out, args.verdicts)
    return apply_accepted(args.out, args.accept, args.data)


if __name__ == "__main__":
    sys.exit(main())
