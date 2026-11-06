"""Section 3.4 vote, audit prompt, part prompt, and part tools.

Equation (2) is a weighted vote of the physics prior, the VLM, and keyword
priors. Unknown-class weights are scaled by 0.4. The paper says the other
weights are source confidence, or a fixed value for keyword priors, and it
does not publish those numbers. The keyword fixed weight is
``taxonomy.DEFAULT_KEYWORD_WEIGHT``, which this tree labels as its own.

CubePart's Quebec crash was a signature mismatch. ``call_qwen_embed_rope``
calls ``forward`` with arguments that match the live signature. It does not
replace that method. Articulate Anything is not downloaded or started here.
A caller supplies the propose and review processes. A review exit of 139, or
any negative signal code, is a ledger fail.
"""

from __future__ import annotations

import inspect
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from sceneagent.agent_loop import append_step, parse_material_class
from sceneagent.gemini_client import PAPER_MODEL_ID
from sceneagent.physics import UNKNOWN_WEIGHT_SCALE, vote_material
from sceneagent.taxonomy import (
    DEFAULT_KEYWORD_WEIGHT,
    MATERIAL_CLASSES,
    keyword_material,
    require_material,
)

# diffusers 0.39: QwenEmbedRope.forward(video_fhw, device=None, max_txt_seq_len=None).
# The CubePart repo calls pos_embed(img_shapes, txt_seq_lens, device=...).
# The second positional binds to device, and the keyword passes device again.
# That is the TypeError. max_txt_seq_len has to travel as a keyword too: the
# 0.39 body raises when it is missing, and a positional text length would
# occupy the device slot.
DIFFUSERS_039_FORWARD = "QwenEmbedRope.forward(video_fhw, device)"
REPO_POS_EMBED_CALL = "pos_embed(img_shapes, txt_seq_lens, device=...)"
ROPE_CRASH = (
    "TypeError: QwenEmbedRope.forward() got multiple values for argument 'device'"
)
# Shells report SIGSEGV as 128 + 11. subprocess reports that signal as a
# negative code, for example -11. Either form is a crashed review.
MOTION_REVIEW_SEGV_EXIT = 139

# Listing from the Quebec zip. The download is not started by this module.
ARTICULATE_UNCOMPRESSED_BYTES = 15_616_309_082
ARTICULATE_SPARE_BYTES = 8 * (1024**3)

HISTORICAL_VLM = (
    {"model_id": "gemini-2.5-flash", "http_status": 404},
    {"model_id": "gemini-3.8-flash", "material_class": "wood", "http_status": 200},
)


def equation2_vote(
    object_name, vlm_class=None, physics_prior=None, vlm_confidence=None
):
    """Vote with equation (2). A missing prior is absent and is not the MLP.

    ``vlm_class`` is the existing class string. ``physics_prior`` is a mapping
    with ``class`` and ``confidence``, or None. ``claimed_mlp`` stays false
    unless the caller sets ``from_mlp`` on a prior this process did not train.
    """
    sources: list[dict[str, Any]] = []
    voters: list[dict[str, Any]] = []
    if physics_prior is None:
        sources.append(
            {
                "source": "physics_prior",
                "status": "absent",
                "class": None,
                "weight": None,
                "claimed_mlp": False,
                "note": "No per-Gaussian MLP output. The class is not the MLP.",
            }
        )
    else:
        material = require_material(physics_prior["class"])
        confidence = float(physics_prior["confidence"])
        from_mlp = bool(physics_prior.get("from_mlp"))
        sources.append(
            {
                "source": "physics_prior",
                "status": "present",
                "class": material,
                "weight": confidence,
                "weight_basis": "caller_confidence",
                "claimed_mlp": from_mlp,
            }
        )
        voters.append(
            {"source": "physics_prior", "class": material, "weight": confidence}
        )
    keyword = keyword_material(object_name, weight=DEFAULT_KEYWORD_WEIGHT)
    keyword["weight_basis"] = (
        "taxonomy.DEFAULT_KEYWORD_WEIGHT; the paper says keyword priors use a "
        "fixed weight and does not publish the number"
    )
    keyword["status"] = "present"
    sources.append(keyword)
    voters.append(
        {"source": "keyword", "class": keyword["class"], "weight": keyword["weight"]}
    )
    if vlm_class is None:
        sources.append(
            {
                "source": "vlm",
                "status": "absent",
                "class": None,
                "weight": None,
                "note": "No VLM class string was supplied.",
            }
        )
    else:
        material = require_material(vlm_class)
        if vlm_confidence is None:
            weight = 1.0
            basis = "fixed_1_because_the_paper_does_not_publish_a_VLM_confidence"
        else:
            weight = float(vlm_confidence)
            basis = "caller_confidence"
        sources.append(
            {
                "source": "vlm",
                "status": "present",
                "class": material,
                "weight": weight,
                "weight_basis": basis,
            }
        )
        voters.append({"source": "vlm", "class": material, "weight": weight})
    winner, scores = vote_material(voters)
    return {
        "equation": "2",
        "unknown_scale": UNKNOWN_WEIGHT_SCALE,
        "object": object_name,
        "material_class": winner,
        "scores": scores,
        "sources": sources,
        "physics_prior": "absent" if physics_prior is None else "present",
        "claimed_mlp": False
        if physics_prior is None
        else bool(physics_prior.get("from_mlp")),
        "class_is_mlp": bool(
            physics_prior and physics_prior.get("from_mlp") and winner is not None
        ),
    }


def audit_prompt(object_name, provisional_class, sources):
    """Text for the second VLM pass. This string does not call the network."""
    lines = [
        "Section 3.4 VLM audit, second pass.",
        f"Object: {object_name}.",
        f"Provisional class from equation (2): {provisional_class}.",
        "Sources:",
    ]
    for source in sources:
        lines.append(
            f"- {source.get('source')}: status {source.get('status')}, "
            f"class {source.get('class')}, weight {source.get('weight')}"
        )
    lines.append(
        "Reply with exactly one published material class, or say the provisional class stands."
    )
    lines.append(
        "The earlier gemini-2.5-flash HTTP 404 and the gemini-3.8-flash class wood "
        "are historical and are not this pass."
    )
    return "\n".join(lines)


def request_audit(object_name, provisional_class, sources, api_key=None, live=False):
    """Return the audit prompt. Call Gemini only when ``live`` is set and a key is present.

    One ``generate_flash`` text call is the cheap call. The default is not to
    call. A live reply is stored separately from the historical 404 and wood class.
    """
    prompt = audit_prompt(object_name, provisional_class, sources)
    body = {
        "prompt": prompt,
        "called": False,
        "model_id": None,
        "http_status": None,
        "text": "",
        "decision": "pending",
        "historical": [dict(item) for item in HISTORICAL_VLM],
        "paper_model_id": PAPER_MODEL_ID,
    }
    if not live or not api_key:
        return body
    from sceneagent.gemini_client import GeminiClient

    reply = GeminiClient(api_key).generate_flash(prompt)
    body["called"] = True
    body["model_id"] = reply.get("model_id")
    body["http_status"] = reply.get("http_status")
    body["text"] = reply.get("text") or ""
    body["decision"] = (
        "replied" if reply.get("http_status") == 200 and body["text"] else "pending"
    )
    body["used_paper_model"] = bool(reply.get("used_paper_model"))
    return body


def part_inspection_prompt(parts):
    """Ask whether hidden geometry is missing. Does not call a model."""
    listed = ", ".join(str(part) for part in parts) if parts else "(no parts)"
    return (
        "Section 3.4 part inspection. A VLM looks at these PartSAM parts and "
        f"decides whether hidden geometry is missing: {listed}. "
        "For each part, say whether hidden geometry is missing, such as a tip "
        "under a cap. Reply yes only if completion is needed. "
        "CubePart must not run unless completion is needed."
    )


def inspect_parts(parts, model_reply=None):
    """Write the part prompt. Without a model reply the decision stays pending.

    CubePart is requested only when the reply says completion is needed.
    """
    prompt = part_inspection_prompt(parts)
    if not model_reply:
        return {
            "prompt": prompt,
            "decision": "pending",
            "completion_needed": None,
            "call_cubepart": False,
            "inspected": False,
            "model_id": None,
        }
    needed = _completion_needed(model_reply)
    return {
        "prompt": prompt,
        "decision": "replied",
        "completion_needed": needed,
        "call_cubepart": needed is True,
        "inspected": True,
        "model_reply": model_reply,
    }


def _completion_needed(text):
    raw = (text or "").strip()
    if not raw:
        return None
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and "completion_needed" in payload:
            return bool(payload["completion_needed"])
    lowered = raw.lower()
    if "completion is needed" in lowered or lowered.startswith("yes"):
        return True
    if "not needed" in lowered or "no missing" in lowered or lowered.startswith("no"):
        return False
    return None


def material_properties_prompt(object_name, notes=""):
    """Ask for the section 3.4 properties the paper names.

    The reply is a class plus numeric friction, mass, and density. This prompt
    does not call the network.
    """
    note = f" {notes}" if notes else ""
    return (
        "Section 3.4 physical properties. "
        f"Object: {object_name}.{note} "
        "Reply with one JSON object and no other text. "
        "Keys: material_class, friction, mass_kg, density_kg_m3. "
        f"material_class must be one of: {', '.join(MATERIAL_CLASSES)}. "
        "friction is one numeric coefficient. mass_kg is kilograms. "
        "density_kg_m3 is kilograms per cubic meter. "
        "Do not sign the reply as a coding agent. "
        "The historical gemini-2.5-flash HTTP 404 and the gemini-3.8-flash class wood "
        "are previous calls, not this reply."
    )


def parse_material_properties(text):
    """Pull a class and three numbers out of a VLM reply. Missing stays None."""
    raw = text or ""
    payload = {}
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            loaded = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, dict):
            payload = loaded

    def number(*keys):
        for key in keys:
            if key not in payload:
                continue
            try:
                value = float(payload[key])
            except (TypeError, ValueError):
                continue
            return value
        return None

    material = payload.get("material_class")
    if material not in MATERIAL_CLASSES:
        material = parse_material_class(str(material or raw))
    friction = number("friction", "friction_static")
    mass = number("mass_kg", "mass")
    density = number("density_kg_m3", "density")
    return {
        "material_class": material,
        "friction": friction,
        "mass_kg": mass,
        "density_kg_m3": density,
        "complete": material is not None and None not in (friction, mass, density),
    }


def request_material_properties(
    object_name, notes="", api_key=None, image_bytes=None, live=False
):
    """Ask Gemini for class, friction, mass, and density. Default is not to call.

    A 404 on gemini-2.5-flash stays the paper attempt. gemini-3.8-flash, when
    chosen by the client, is recorded as the fallback and is not labeled 2.5.
    """
    prompt = material_properties_prompt(object_name, notes)
    body = {
        "prompt": prompt,
        "called": False,
        "model_id": None,
        "http_status": None,
        "paper_model_id": PAPER_MODEL_ID,
        "used_paper_model": False,
        "model_role": None,
        "fallback_model_id": None,
        "text": "",
        "properties": None,
        "historical": [dict(item) for item in HISTORICAL_VLM],
    }
    if not live or not api_key:
        return body
    from sceneagent.gemini_client import GeminiClient

    reply = GeminiClient(api_key).generate_flash(prompt, image_bytes=image_bytes)
    body["called"] = True
    body["model_id"] = reply.get("model_id")
    body["http_status"] = reply.get("http_status")
    body["paper_http_status"] = reply.get("paper_http_status")
    body["used_paper_model"] = bool(reply.get("used_paper_model"))
    body["fallback_model_id"] = reply.get("fallback_model_id")
    body["model_role"] = "paper" if reply.get("used_paper_model") else "fallback"
    body["text"] = reply.get("text") or ""
    body["properties"] = parse_material_properties(body["text"])
    if api_key:
        body["prompt"] = body["prompt"].replace(api_key, "[redacted]")
        body["text"] = body["text"].replace(api_key, "[redacted]")
    return body


def partsam_summary_prompt(masks, parts_after_nms, parts):
    """Text summary of PartSAM parts. Does not call CubePart or a model."""
    listed = []
    for part in parts or []:
        if isinstance(part, dict):
            listed.append(
                f"index {part.get('index')} rgb {part.get('rgb')} faces {part.get('faces')}"
            )
        else:
            listed.append(str(part))
    body = (
        f"PartSAM summary: {int(masks)} masks, {int(parts_after_nms)} parts after NMS. "
        + ("; ".join(listed) if listed else "no per-part colors")
    )
    return part_inspection_prompt([body])


def choose_qwen_embed_rope_call(parameter_names):
    """Pick the call shape from the live ``forward`` parameter names.

    ``('video_fhw', 'device')`` is diffusers 0.39. ``max_txt_seq_len`` is
    included when that parameter is on the same signature. A signature that
    still takes ``txt_seq_lens`` keeps the text length positional and passes
    ``device`` by keyword, which is the call that 0.39 rejects.
    """
    names = tuple(name for name in parameter_names if name != "self")
    if "txt_seq_lens" in names and ("video_fhw" in names or "img_shapes" in names):
        shape = "img_shapes" if "img_shapes" in names else "video_fhw"
        keyword = ("device",) if "device" in names else ()
        call = f"forward({shape}, txt_seq_lens"
        if "device" in names:
            call += ", device=device"
        call += ")"
        return {
            "signature": REPO_POS_EMBED_CALL
            if shape == "img_shapes"
            else "QwenEmbedRope.forward(video_fhw, txt_seq_lens, device)",
            "positional": (shape, "txt_seq_lens"),
            "keyword": keyword,
            "call": call,
        }
    if names and names[0] == "video_fhw":
        keyword_names = []
        pieces = ["video_fhw"]
        if "device" in names:
            keyword_names.append("device")
            pieces.append("device=device")
        if "max_txt_seq_len" in names:
            keyword_names.append("max_txt_seq_len")
            pieces.append("max_txt_seq_len=max_txt_seq_len")
        return {
            "signature": DIFFUSERS_039_FORWARD,
            "positional": ("video_fhw",),
            "keyword": tuple(keyword_names),
            "call": "forward(" + ", ".join(pieces) + ")",
        }
    raise ValueError(f"unrecognized QwenEmbedRope parameters: {names}")


def call_qwen_embed_rope(pos_embed, img_shapes, txt_seq_lens, device):
    """Call ``QwenEmbedRope.forward`` with the live signature's arguments.

    Does not assign ``forward``. On diffusers 0.39 the text length is
    ``max_txt_seq_len`` and it is passed by keyword, next to ``device``.
    """
    forward = getattr(pos_embed, "forward", None)
    if not callable(forward):
        forward = pos_embed
    names = _forward_parameter_names(forward)
    spec = choose_qwen_embed_rope_call(names)
    positional = [
        _rope_argument(name, img_shapes, txt_seq_lens, device)
        for name in spec["positional"]
    ]
    keyword = {
        name: _rope_argument(name, img_shapes, txt_seq_lens, device)
        for name in spec["keyword"]
    }
    return forward(*positional, **keyword)


def _forward_parameter_names(forward):
    try:
        signature = inspect.signature(forward)
    except (TypeError, ValueError) as exc:
        raise TypeError("QwenEmbedRope.forward has no inspectable signature") from exc
    names = []
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        names.append(name)
    return tuple(names)


def _rope_argument(name, img_shapes, txt_seq_lens, device):
    if name in ("video_fhw", "img_shapes"):
        return img_shapes
    if name == "txt_seq_lens":
        return txt_seq_lens
    if name == "device":
        return device
    if name == "max_txt_seq_len":
        return _max_txt_seq_len(txt_seq_lens)
    raise ValueError(f"unrecognized QwenEmbedRope parameter: {name}")


def _max_txt_seq_len(txt_seq_lens):
    """The longest text sequence. 0.39 takes that integer, not the list."""
    if isinstance(txt_seq_lens, int):
        return txt_seq_lens
    lengths = [int(item) for item in txt_seq_lens]
    if not lengths:
        raise ValueError(
            "txt_seq_lens is empty; diffusers 0.39 requires max_txt_seq_len"
        )
    return max(lengths)


def cubepart_status(part_mesh_path=None):
    """Fixed only when a part mesh file exists. The RoPE note is not a mesh."""
    mesh = Path(part_mesh_path) if part_mesh_path else None
    has_mesh = bool(mesh and mesh.is_file() and mesh.stat().st_size > 0)
    return {
        "status": "fixed" if has_mesh else "still_open",
        "part_mesh": str(mesh) if has_mesh else None,
        "gpu_rerun": False,
        "compat_note": (
            f"{DIFFUSERS_039_FORWARD} versus the repo call {REPO_POS_EMBED_CALL}. "
            f"The Quebec failure was {ROPE_CRASH}."
        ),
    }


def articulate_download_allowed(
    gpu_up, free_bytes, uncompressed_bytes=ARTICULATE_UNCOMPRESSED_BYTES
):
    """True only when a GPU is up and free space exceeds the extract plus 8 GiB."""
    if not gpu_up:
        return False
    return int(free_bytes) > int(uncompressed_bytes) + ARTICULATE_SPARE_BYTES


def articulate_row(gpu_up, free_bytes, urdf_path=None):
    """Leave the row open until a URDF file exists. Does not download the zip."""
    urdf = Path(urdf_path) if urdf_path else None
    has_urdf = bool(urdf and urdf.is_file() and urdf.stat().st_size > 0)
    allowed = articulate_download_allowed(gpu_up, free_bytes)
    return {
        "status": "fixed" if has_urdf else "still_open",
        "download_started": False,
        "download_allowed": allowed,
        "urdf": str(urdf) if has_urdf else None,
        "reason": "URDF exists."
        if has_urdf
        else "No URDF. The Articulate Anything download was not started.",
    }


def cubepart_mesh_result(part_mesh_path):
    """Tool result for a CubePart mesh. A missing file is not a pass."""
    mesh = cubepart_status(part_mesh_path)["part_mesh"]
    return {
        "ok": mesh is not None,
        "artifact": mesh,
        "reason": "" if mesh else "CubePart wrote no part mesh.",
        "detail": {"part_mesh": mesh, "kind": "cubepart_mesh"},
    }


def partsam_link_names(parts):
    """URDF link tokens for PartSAM parts. Empty input is refused."""
    names = []
    seen: dict[str, int] = {}
    for index, part in enumerate(parts or []):
        token = _link_token(_part_raw_name(part, index))
        count = seen.get(token, 0)
        seen[token] = count + 1
        if count:
            token = f"{token}_{count}"
        names.append(token)
    if not names:
        raise ValueError(
            "PartSAM parts are empty. Articulate Anything was not started."
        )
    return names


def articulate_propose_command(spec_path, out_dir):
    """Command the GPU step would run from an Articulate Anything checkout.

    This string is recorded. It is not executed here.
    """
    return [
        "python",
        "articulate.py",
        f"prompt={spec_path}",
        "modality=text",
        f"out_dir={out_dir}",
    ]


def motion_review_command(urdf_path, review_dir):
    """Headless ``sapien_simulate`` of every joint. Not executed here."""
    return [
        "python",
        "articulate_anything/physics/sapien_simulate.py",
        f"urdf.file={urdf_path}",
        f"output.dir={review_dir}",
        "headless=true",
        "simulation_params.stationary_or_move=move",
        "simulation_params.joint_name=all",
    ]


def classify_motion_review(returncode, stdout="", stderr=""):
    """Map a review process to accept, reject, or fail.

    Exit 139 and any negative code mean the process was killed. Those are
    fail even when the captured text says the motion was good. Exit 0 is
    accept only when the text says so. Every other finished process is reject.
    """
    if returncode is None or returncode < 0 or returncode == MOTION_REVIEW_SEGV_EXIT:
        return "fail"
    if returncode == 0 and _motion_verdict(stdout, stderr) == "accept":
        return "accept"
    return "reject"


def run_partsam_articulation(out_dir, parts, part_mesh=None, propose=None, review=None):
    """Propose a URDF for these PartSAM parts, then review the motion.

    ``propose`` and ``review`` are the processes that stand in for Articulate
    Anything and ``sapien_simulate``. Omitting them does not start either
    program, and that hold is not an accept. A review crash is fail.

    ``propose`` and ``review`` each receive one dict and return an object
    with ``returncode``, ``stdout``, and ``stderr``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = partsam_link_names(parts)
    spec_path = out / "partsam_parts.json"
    urdf_path = out / "mobility.urdf"
    review_dir = out / "motion_review"
    spec_path.write_text(
        json.dumps(
            {
                "source": "partsam",
                "parts": [{"name": name} for name in names],
                "note": (
                    "Propose joints for these PartSAM links. "
                    "A retrieved PartNet table is not this object."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    propose_command = articulate_propose_command(spec_path, out)
    review_command = motion_review_command(urdf_path, review_dir)
    mesh = cubepart_status(part_mesh)["part_mesh"] if part_mesh else None
    entries = []
    if part_mesh is not None:
        entries.append(append_step(out, "cubepart", cubepart_mesh_result(part_mesh)))
    outcome = _articulation_outcome(
        names,
        urdf_path,
        spec_path,
        propose,
        review,
        propose_command,
        review_command,
        review_dir,
    )
    detail = {
        "part_mesh": mesh,
        "urdf": str(urdf_path) if urdf_path.is_file() else None,
        "review": outcome["review"],
        "returncode": outcome["returncode"],
        "links_match": outcome["links_match"],
        "propose_command": propose_command,
        "review_command": review_command,
        "kind": "partsam_motion_review",
    }
    result = {
        "ok": outcome["ok"],
        "artifact": detail["urdf"],
        "reason": outcome["reason"],
        "detail": detail,
    }
    entries.append(append_step(out, "articulate", result))
    return {
        "entries": entries,
        "urdf": detail["urdf"],
        "review": outcome["review"],
        "part_mesh": mesh,
    }


def _articulation_outcome(
    names,
    urdf_path,
    spec_path,
    propose,
    review,
    propose_command,
    review_command,
    review_dir,
):
    if propose is None or review is None:
        return {
            "ok": False,
            "review": "reject",
            "returncode": None,
            "links_match": False,
            "reason": "Articulate Anything was not started. No motion review ran.",
        }
    proposed = _invoke(
        propose,
        {
            "command": propose_command,
            "spec_path": spec_path,
            "urdf_path": urdf_path,
            "parts": names,
        },
    )
    propose_code, _stdout, stderr = _completed_io(proposed)
    if _crashed(propose_code):
        return {
            "ok": False,
            "review": "fail",
            "returncode": propose_code,
            "links_match": False,
            "reason": (
                f"Articulate Anything crashed with return code {propose_code} "
                "before the motion review. A crash is fail, not a pass."
            ),
        }
    if not urdf_path.is_file() or urdf_path.stat().st_size <= 0:
        tail = (stderr or "").strip()
        reason = "Articulate Anything wrote no URDF."
        if tail:
            reason = f"{reason} {tail}"
        return {
            "ok": False,
            "review": "reject",
            "returncode": propose_code,
            "links_match": False,
            "reason": reason,
        }
    reviewed = _invoke(
        review,
        {
            "command": review_command,
            "urdf_path": urdf_path,
            "review_dir": review_dir,
        },
    )
    code, stdout, review_err = _completed_io(reviewed)
    verdict = classify_motion_review(code, stdout, review_err)
    links_match = _links_match(urdf_path, names)
    if verdict == "fail":
        return {
            "ok": False,
            "review": "fail",
            "returncode": code,
            "links_match": links_match,
            "reason": (
                f"Motion review crashed with return code {code}. "
                "Exit 139 or a signal is fail, not a pass."
            ),
        }
    if verdict == "accept" and links_match:
        return {
            "ok": True,
            "review": "accept",
            "returncode": code,
            "links_match": True,
            "reason": "",
        }
    if not links_match:
        return {
            "ok": False,
            "review": "reject",
            "returncode": code,
            "links_match": False,
            "reason": (
                "URDF links are not the PartSAM parts. "
                f"The motion review said {verdict}. That URDF is not accepted."
            ),
        }
    return {
        "ok": False,
        "review": "reject",
        "returncode": code,
        "links_match": True,
        "reason": "Motion review rejected the URDF.",
    }


def _invoke(func, payload):
    try:
        return func(payload)
    except Exception as exc:
        return _FailedProcess(exc)


def _crashed(returncode):
    return classify_motion_review(returncode) == "fail"


def _links_match(urdf_path, names):
    try:
        links = urdf_link_names(urdf_path)
    except ET.ParseError:
        return False
    return set(names) <= set(links)


def urdf_link_names(path):
    """Link names from a URDF. Namespace tags count as links too."""
    names = []
    for node in ET.parse(path).getroot().iter():
        tag = node.tag.rsplit("}", 1)[-1]
        if tag != "link":
            continue
        name = node.get("name")
        if name:
            names.append(name)
    return names


def _completed_io(completed):
    if completed is None:
        return None, "", ""
    code = getattr(completed, "returncode", None)
    stdout = getattr(completed, "stdout", "") or ""
    stderr = getattr(completed, "stderr", "") or ""
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", "replace")
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", "replace")
    if code is not None:
        code = int(code)
    return code, stdout, stderr


def _motion_verdict(stdout, stderr):
    payload = _json_object(stdout) or _json_object(stderr)
    if not isinstance(payload, dict):
        return None
    if "feedback_score" in payload:
        try:
            score = float(payload["feedback_score"])
        except (TypeError, ValueError):
            return "reject"
        if score < 0:
            return "reject"
    decision = str(payload.get("decision") or "").strip().lower()
    if decision in {"accept", "pass"}:
        return "accept"
    if decision in {"reject", "fail"}:
        return "reject"
    reason = str(payload.get("failure_reason") or "").strip().lower()
    if reason == "success":
        return "accept"
    if reason:
        return "reject"
    return None


def _json_object(text):
    raw = text or ""
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _part_raw_name(part, index):
    if isinstance(part, dict):
        raw = part.get("name") or part.get("id")
        if raw:
            return str(raw)
    elif part:
        return str(part)
    return f"partsam_{index:02d}"


def _link_token(raw):
    chars = []
    for char in str(raw).strip().lower():
        if char.isalnum() or char == "_":
            chars.append(char)
        else:
            chars.append("_")
    token = "".join(chars).strip("_")
    return token or "part"


class _FailedProcess:
    """A propose or review callable that raised. That is not a clean exit."""

    def __init__(self, exc):
        self.returncode = None
        self.stdout = ""
        self.stderr = f"{type(exc).__name__}: {exc}"
