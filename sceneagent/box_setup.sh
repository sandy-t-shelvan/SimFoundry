#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Installs the SceneAgent GPU stack on an empty Vast box (gpu-smoke `setup`).
#
#   bash box_setup.sh run    OUT   # run every stage not yet done (detached by the caller)
#   bash box_setup.sh report OUT   # write OUT/setup_evidence.json and OUT/ledger.json
#
# Paths mirror the box this pipeline ran on before (58301075, see
# docs/session-handoff.md), so the pipeline code's interpreter and checkpoint
# constants hold unchanged. Each stage is idempotent: it writes
# /workspace/setup/state/<stage>.ok when it succeeds and is skipped after
# that, so a dropped runner or a re-dispatch resumes where it stopped.
# Logs go to /workspace/setup/logs/<stage>.log. They are not published: the
# report carries only stage results, versions, commits and checksums.
#
# Versions are the old box's, except that GroundingDINO's env uses torch
# 2.5.1+cu121 (its compiled extension must match this box's CUDA 12.1
# toolkit) and so does the main `sa` env. COLMAP 4.1.1 (only for the mapper
# comparison) is not installed.
#
# The Hugging Face token for SAM 3D's gated weights is written to
# /root/.cache/huggingface/token by the runner before this runs (Sandeep,
# 2026-11-11). This script never prints it.

set -uo pipefail

MODE="${1:-}"
OUT="${2:-/workspace/sf4-smoke/setup}"
ROOT=/workspace/setup
STATE="$ROOT/state"
LOGS="$ROOT/logs"
MAMBA=/opt/conda/bin/mamba
ENVS=/root/.local/share/mamba/envs
TORCH_CU121=https://download.pytorch.org/whl/cu121
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
export DEBIAN_FRONTEND=noninteractive
export PIP_NO_INPUT=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export MAX_JOBS=32

COLMAP_TAG=3.11.1
GSAM_ROOT=/root/SceneagentX/third_party/Grounded-SAM-2
GDINO_URL=https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
SAM_URL=https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
SAM_WEIGHTS=/workspace/models/sam/sam_vit_h_4b8939.pth
# sam2_user.py reads SAM2_CHECKPOINT. The 2.1 file is the checkout's other large
# checkpoint. Both hashes are the files fetched on 60254439.
SAM2_URL=https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
SAM2_1_URL=https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
SAM2_CHECKPOINT="$GSAM_ROOT/checkpoints/sam2_hiera_large.pt"
SAM2_1_CHECKPOINT="$GSAM_ROOT/checkpoints/sam2.1_hiera_large.pt"
SAM2_SHA256=7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b
SAM2_1_SHA256=2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318
DA3_ROOT=/workspace/src/depth-anything-3
SAM3D_ROOT=/root/sam-3d-objects
SAM3D_ENV=/opt/conda/envs/sam3d10
# HEAD of the checkout that finished s_sam3d on 60254439.
SAM3D_COMMIT=f91db411c50efee93d8db7aeb323885650f6f722
# Isaac Lab v2.3.2's pip page installs Isaac Sim 5.1.0 into its own env.
# That is not the simfoundry env, and usd-core is not installed into either.
ISAACSIM_PIP=5.1.0
ISAACLAB_TAG=v2.3.2
ISAAC_ENV="$ENVS/isaaclab"
ISAACLAB_ROOT=/workspace/src/IsaacLab
STAGES="apt colmap env_sa da3 env_nerf env_gsam2 sam_weights sam3d sam3d_weights isaaclab"

mkdir -p "$STATE" "$LOGS" "$OUT"

clone() {  # clone URL DIR [TAG]: shallow, once
  if [ ! -d "$2/.git" ]; then
    if [ -n "${3:-}" ]; then
      git clone --depth 1 --branch "$3" "$1" "$2"
    else
      git clone --depth 1 "$1" "$2"
    fi
  fi
}

fetch_sha256() {  # URL DEST SHA256: keep a file that already matches
  local url="$1" dest="$2" want="$3" got
  mkdir -p "$(dirname "$dest")"
  if [ -s "$dest" ]; then
    got=$(sha256sum "$dest" | awk '{print $1}')
    if [ "$got" = "$want" ]; then
      return 0
    fi
    rm -f "$dest"
  fi
  wget -q -O "$dest.partial" "$url"
  mv "$dest.partial" "$dest"
  got=$(sha256sum "$dest" | awk '{print $1}')
  if [ "$got" != "$want" ]; then
    echo "checksum mismatch for $dest" >&2
    return 1
  fi
}

s_apt() {
  apt-get update
  apt-get install -y --no-install-recommends \
    git git-lfs rsync tmux ffmpeg p7zip-full wget curl ca-certificates \
    build-essential cmake ninja-build pkg-config \
    libboost-program-options-dev libboost-graph-dev libboost-system-dev \
    libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev \
    libgoogle-glog-dev libgflags-dev libgtest-dev libgmock-dev \
    libsqlite3-dev libglew-dev qtbase5-dev libqt5opengl5-dev \
    libcgal-dev libceres-dev libgl1 libglib2.0-0
  git lfs install --system
}

s_colmap() {
  local src=/workspace/src/colmap-$COLMAP_TAG
  clone https://github.com/colmap/colmap "$src" "$COLMAP_TAG"
  cmake -S "$src" -B "$src/build" -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/colmap-cuda \
    -DCUDA_ENABLED=ON \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
    -DGUI_ENABLED=OFF
  ninja -C "$src/build" -j "$MAX_JOBS" install
  /opt/colmap-cuda/bin/colmap -h | head -3
  if /opt/colmap-cuda/bin/colmap -h | grep -q "without CUDA"; then
    echo "COLMAP was built without CUDA" >&2
    return 1
  fi
}

s_env_sa() {
  [ -x "$ENVS/sa/bin/python3" ] || "$MAMBA" create -y -p "$ENVS/sa" python=3.11
  "$ENVS/sa/bin/pip" install torch==2.5.1 torchvision==0.20.1 --index-url "$TORCH_CU121"
  # The pins of requirements-sceneagent-ci.txt, so the box runs what CI tests.
  "$ENVS/sa/bin/pip" install numpy==2.4.6 scipy==1.16.3 trimesh==5.2.0 \
    pillow==13.0.0 py7zr==1.1.4 huggingface_hub
  # 4.x requires torch>=2.6 and replaces the pin above (found on 60254439).
  "$ENVS/sa/bin/pip" install open_clip_torch==3.3.0
  "$ENVS/sa/bin/python3" -c "import numpy, scipy, trimesh, PIL, py7zr, torch, open_clip; assert torch.cuda.is_available() and torch.__version__.startswith('2.5.1')"
}

s_da3() {
  clone https://github.com/ByteDance-Seed/Depth-Anything-3 "$DA3_ROOT"
  # Its dependencies only; bin_scale puts the checkout's src on PYTHONPATH.
  "$ENVS/sa/bin/pip" install -e "$DA3_ROOT"
  # DA3's pyproject asks for numpy<2 and pip downgrades it; CI's 2.4.6 still
  # imports DA3, and the pipeline is tested against it, so put it back.
  "$ENVS/sa/bin/pip" install numpy==2.4.6
  cd /tmp
  PYTHONPATH="$DA3_ROOT/src" "$ENVS/sa/bin/python3" -c "import depth_anything_3"
  # The weights bin_scale loads, into the cache it reads (HF_HUB_CACHE).
  HF_HUB_CACHE=/workspace/sf4-smoke/hf-cache "$ENVS/sa/bin/python3" -c \
    "from huggingface_hub import snapshot_download; snapshot_download('depth-anything/DA3METRIC-LARGE')"
}

s_env_nerf() {
  [ -x "$ENVS/nerf/bin/python3" ] || "$MAMBA" create -y -p "$ENVS/nerf" python=3.10
  "$ENVS/nerf/bin/pip" install torch==2.1.2 torchvision==0.16.2 --index-url "$TORCH_CU121"
  "$ENVS/nerf/bin/pip" install ninja nerfstudio==1.1.4 gsplat==1.0.0
  # torch 2.1.2's extension builder imports pkg_resources, which setuptools
  # 82 removed. nerfstudio pulls that newer setuptools, so the pin has to
  # come after it or gsplat never compiles (found on 60254439).
  "$ENVS/nerf/bin/pip" install "setuptools>=69,<81"
  # gsplat compiles its kernels on first use; do it here, not in the first splat run.
  cd /tmp
  "$ENVS/nerf/bin/python3" -c "import gsplat, torch; from gsplat.cuda._backend import _C; assert _C is not None and torch.cuda.is_available()"
  "$ENVS/nerf/bin/ns-train" --help > /dev/null
}

s_env_gsam2() {
  [ -x "$ENVS/gsam2/bin/python3" ] || "$MAMBA" create -y -p "$ENVS/gsam2" python=3.10
  "$ENVS/gsam2/bin/pip" install torch==2.5.1 torchvision==0.20.1 --index-url "$TORCH_CU121"
  mkdir -p "$(dirname "$GSAM_ROOT")"
  clone https://github.com/IDEA-Research/Grounded-SAM-2 "$GSAM_ROOT"
  cd "$GSAM_ROOT"
  SAM2_BUILD_ALLOW_ERRORS=1 "$ENVS/gsam2/bin/pip" install -e .
  "$ENVS/gsam2/bin/pip" install --no-build-isolation -e grounding_dino
  # GroundingDINO's runtime imports; transformers 5.x refuses torch 2.5.1.
  "$ENVS/gsam2/bin/pip" install opencv-python-headless pycocotools supervision \
    addict yapf timm "transformers>=4.46,<5"
  mkdir -p gdino_checkpoints
  [ -s gdino_checkpoints/groundingdino_swint_ogc.pth ] || \
    wget -q -O gdino_checkpoints/groundingdino_swint_ogc.pth "$GDINO_URL"
  fetch_sha256 "$SAM2_URL" "$SAM2_CHECKPOINT" "$SAM2_SHA256"
  fetch_sha256 "$SAM2_1_URL" "$SAM2_1_CHECKPOINT" "$SAM2_1_SHA256"
  # sam2_user.py does not search this checkout; later runs export this file.
  printf 'export SAM2_CHECKPOINT=%q\n' "$SAM2_CHECKPOINT" > "$ROOT/sam2.env"
  cd /tmp
  PYTHONPATH="$GSAM_ROOT" "$ENVS/gsam2/bin/python3" -c \
    "from grounding_dino.groundingdino import _C"
  # Import the CUDA op once, under one name: loading the same .so again as
  # groundingdino._C aborts the interpreter on exit (found on 60254439).
}

s_sam_weights() {
  mkdir -p "$(dirname "$SAM_WEIGHTS")"
  [ -s "$SAM_WEIGHTS" ] || wget -q -O "$SAM_WEIGHTS" "$SAM_URL"
  sha256sum "$SAM_WEIGHTS"
}

s_sam3d() {
  clone https://github.com/facebookresearch/sam-3d-objects "$SAM3D_ROOT"
  if [ "$(git -C "$SAM3D_ROOT" rev-parse HEAD)" != "$SAM3D_COMMIT" ]; then
    git -C "$SAM3D_ROOT" fetch --depth 1 origin "$SAM3D_COMMIT"
    git -C "$SAM3D_ROOT" checkout --detach "$SAM3D_COMMIT"
  fi
  cd "$SAM3D_ROOT"
  [ -x "$SAM3D_ENV/bin/python3" ] || \
    "$MAMBA" env create -y -f environments/default.yml -p "$SAM3D_ENV"
  local pip="$SAM3D_ENV/bin/pip"
  export CONDA_PREFIX="$SAM3D_ENV"
  export PATH="$SAM3D_ENV/bin:$PATH"
  export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com $TORCH_CU121"
  # The dev extra pulls unpinned pipdeptree; 4.x depends on nab-* packages and
  # pip backtracks through versions without end (found on 60254439).
  echo "pipdeptree==2.28.0" > "$ROOT/sam3d-constraints.txt"
  export PIP_CONSTRAINT="$ROOT/sam3d-constraints.txt"
  "$pip" install -e '.[dev]'
  "$pip" install -e '.[p3d]'
  PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html" \
    "$pip" install -e '.[inference]'
  if [ -x ./patching/hydra ]; then ./patching/hydra; fi
  cd notebook
  "$SAM3D_ENV/bin/python3" -c "import inference"
}

s_sam3d_weights() {
  if [ ! -s /root/.cache/huggingface/token ]; then
    echo "no Hugging Face token on the box; add the HF_TOKEN Actions secret and re-run setup" >&2
    return 1
  fi
  cd "$SAM3D_ROOT"
  if [ ! -s checkpoints/hf/pipeline.yaml ]; then
    "$SAM3D_ENV/bin/pip" install 'huggingface-hub[cli]<1.0'
    "$SAM3D_ENV/bin/hf" download --repo-type model --local-dir checkpoints/hf-download \
      --max-workers 1 facebook/sam-3d-objects
    rm -rf checkpoints/hf
    mv checkpoints/hf-download/checkpoints checkpoints/hf
    rm -rf checkpoints/hf-download
  fi
  test -s checkpoints/hf/pipeline.yaml
}

s_isaaclab() {
  # Pip Isaac Sim needs glibc 2.35. Ubuntu 20.04's 2.31 fails the wheel
  # lookup, so stop before the download.
  local glibc
  glibc=$(ldd --version | awk 'NR==1 {print $NF}')
  /opt/conda/bin/python3 -c 'import sys; got=tuple(int(p) for p in sys.argv[1].split(".")[:2]); need=(2, 35); sys.exit(0 if got >= need else 1)' "$glibc" \
    || { echo "glibc $glibc is below 2.35; Isaac Sim pip wheels will not install" >&2; return 1; }
  [ -x "$ISAAC_ENV/bin/python3" ] || "$MAMBA" create -y -p "$ISAAC_ENV" python=3.11
  "$ISAAC_ENV/bin/pip" install --upgrade pip
  "$ISAAC_ENV/bin/pip" install "isaacsim[all,extscache]==$ISAACSIM_PIP" \
    --extra-index-url https://pypi.nvidia.com
  if [ ! -d "$ISAACLAB_ROOT/.git" ]; then
    git clone --depth 1 --branch "$ISAACLAB_TAG" https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_ROOT"
  fi
  if [ "$(git -C "$ISAACLAB_ROOT" rev-parse --abbrev-ref HEAD)" != "$ISAACLAB_TAG" ] \
    && ! git -C "$ISAACLAB_ROOT" describe --tags --exact-match 2>/dev/null | grep -qx "$ISAACLAB_TAG"; then
    git -C "$ISAACLAB_ROOT" fetch --depth 1 origin "refs/tags/$ISAACLAB_TAG:refs/tags/$ISAACLAB_TAG"
    git -C "$ISAACLAB_ROOT" checkout --detach "$ISAACLAB_TAG"
  fi
  # The installer prompts for the Omniverse EULA. The documented variable
  # accepts YES and skips that prompt.
  export OMNI_KIT_ACCEPT_EULA=YES
  export PATH="$ISAAC_ENV/bin:$PATH"
  cd "$ISAACLAB_ROOT"
  ./isaaclab.sh --install
  du -sb "$ISAACLAB_ROOT" "$ISAAC_ENV" | tee "$ROOT/isaaclab-size.txt"
  local here
  here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  OMNI_KIT_ACCEPT_EULA=YES "$ISAAC_ENV/bin/python3" "$here/isaaclab_headless_check.py"
}

run_stage() {
  local name="$1"
  if [ -f "$STATE/$name.ok" ]; then
    echo "$name: already done"
    return 0
  fi
  echo "$name: started $(date -u +%FT%TZ)"
  # Not inside an `if`: bash ignores `set -e` in a conditional's subshell.
  ( set -e; "s_$name" ) > "$LOGS/$name.log" 2>&1
  local code=$?
  if [ "$code" -eq 0 ]; then
    date -u +%FT%TZ > "$STATE/$name.ok"
    rm -f "$STATE/$name.failed"
    echo "$name: ok"
  else
    echo "$code" > "$STATE/$name.failed"
    echo "$name: failed with exit $code"
  fi
  return "$code"
}

do_run() {
  exec 9> "$ROOT/lock"
  if ! flock -n 9; then
    echo "setup already running"
    return 0
  fi
  rm -f "$STATE/done"
  for name in $STAGES; do
    run_stage "$name"
  done
  date -u +%FT%TZ > "$STATE/done"
}

do_report() {
  local py=/opt/conda/bin/python3
  STATE="$STATE" LOGS="$LOGS" OUT="$OUT" STAGES="$STAGES" ENVS="$ENVS" \
    GSAM_ROOT="$GSAM_ROOT" DA3_ROOT="$DA3_ROOT" SAM3D_ROOT="$SAM3D_ROOT" \
    SAM3D_ENV="$SAM3D_ENV" SAM_WEIGHTS="$SAM_WEIGHTS" COLMAP_TAG="$COLMAP_TAG" \
    SAM2_CHECKPOINT="$SAM2_CHECKPOINT" SAM2_1_CHECKPOINT="$SAM2_1_CHECKPOINT" \
    ISAAC_ENV="$ISAAC_ENV" ISAACLAB_ROOT="$ISAACLAB_ROOT" \
    "$py" - <<'PY'
import hashlib, json, os, pathlib, re, subprocess

env = os.environ
# The notebook sets this before importing the package. Without it the import
# looks for a submodule the checkout does not ship, and the report would
# record a failure for an install the stage already accepted.
env["LIDRA_SKIP_INIT"] = "true"
state = pathlib.Path(env["STATE"])
logs = pathlib.Path(env["LOGS"])
out = pathlib.Path(env["OUT"])
stages = env["STAGES"].split()
# A log line may carry a URL or a path; never anything that looks like a token.
SECRETISH = re.compile(r"hf_[A-Za-z0-9]{20,}|token|secret|passw", re.I)


def tail(name, n=6):
    path = logs / f"{name}.log"
    if not path.is_file():
        return []
    lines = path.read_text(errors="replace").splitlines()[-n:]
    return [l[-240:] for l in lines if not SECRETISH.search(l)]


def run(argv, timeout=180):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return (p.stdout or p.stderr).strip()
    except Exception as exc:  # recorded, not raised
        return f"unavailable: {type(exc).__name__}"


def pyver(python, mods):
    code = (
        "import importlib, json, sys\n"
        "v = {'python': sys.version.split()[0]}\n"
        f"for m in {mods!r}:\n"
        "    try:\n"
        "        mod = importlib.import_module(m)\n"
        "        v[m] = getattr(mod, '__version__', 'installed')\n"
        "    except Exception as e:\n"
        "        v[m] = 'MISSING: ' + type(e).__name__\n"
        "try:\n"
        "    import torch\n"
        "    v['torch_cuda'] = torch.version.cuda\n"
        "    v['cuda_available'] = torch.cuda.is_available()\n"
        "except Exception:\n"
        "    pass\n"
        "print(json.dumps(v))\n"
    )
    if not os.access(python, os.X_OK):
        return {"python": "MISSING"}
    text = run([python, "-c", code], timeout=300)
    try:
        return json.loads(text.splitlines()[-1])
    except Exception:
        return {"error": text[-200:]}


def commit(path):
    return run(["git", "-C", path, "rev-parse", "HEAD"]) if pathlib.Path(path, ".git").is_dir() else None


def sha256(path):
    p = pathlib.Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def running_stage():
    """The stage that has a log but no result yet: what the install is doing now."""
    for name in stages:
        if (logs / f"{name}.log").is_file() and not (state / f"{name}.ok").is_file() and not (state / f"{name}.failed").is_file():
            return name
    return None


results = {}
for name in stages:
    if (state / f"{name}.ok").is_file():
        results[name] = {"status": "ok", "at": (state / f"{name}.ok").read_text().strip()}
    elif (state / f"{name}.failed").is_file():
        results[name] = {"status": "failed", "exit": (state / f"{name}.failed").read_text().strip(), "log_tail": tail(name)}
    elif name == running_stage():
        results[name] = {"status": "running", "log_tail": tail(name, 12)}
    else:
        results[name] = {"status": "pending"}
envs = env["ENVS"]
sam3d_py = env["SAM3D_ENV"] + "/bin/python3"
versions = {
    "gpu": run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
    "nvcc": run(["/usr/local/cuda-12.1/bin/nvcc", "--version"]).splitlines()[-1:],
    "colmap": run(["/opt/colmap-cuda/bin/colmap", "-h"]).splitlines()[:2],
    "ffmpeg": run(["ffmpeg", "-version"]).splitlines()[:1],
    "sa": pyver(envs + "/sa/bin/python3", ["numpy", "scipy", "trimesh", "PIL", "py7zr", "torch", "torchvision", "open_clip"]),
    "nerf": pyver(envs + "/nerf/bin/python3", ["torch", "torchvision", "nerfstudio", "gsplat"]),
    "gsam2": pyver(envs + "/gsam2/bin/python3", ["torch", "torchvision", "groundingdino", "sam2", "transformers"]),
    "sam3d10": pyver(sam3d_py, ["torch", "pytorch3d", "kaolin", "moge", "sam3d_objects", "gsplat"]),
    "isaaclab": pyver(env["ISAAC_ENV"] + "/bin/python3", ["isaacsim", "isaaclab"]),
    "isaaclab_bytes": run(["du", "-sb", env["ISAACLAB_ROOT"], env["ISAAC_ENV"]]),
}
document = {
    "box_setup": "ours: sceneagent/box_setup.sh",
    "stages": results,
    "all_ok": all(r["status"] == "ok" for r in results.values()),
    "finished": (state / "done").is_file(),
    # Stage start/ok/failed lines only (run_stage writes nothing else here).
    "run_log": [l for l in pathlib.Path("/workspace/setup/run.log").read_text(errors="replace").splitlines()[-30:] if not SECRETISH.search(l)] if pathlib.Path("/workspace/setup/run.log").is_file() else [],
    "versions": versions,
    "commits": {
        "colmap": env["COLMAP_TAG"],
        "grounded_sam_2": commit(env["GSAM_ROOT"]),
        "depth_anything_3": commit(env["DA3_ROOT"]),
        "sam_3d_objects": commit(env["SAM3D_ROOT"]),
    },
    "checksums": {
        "sam_vit_h_4b8939.pth": sha256(env["SAM_WEIGHTS"]),
        "groundingdino_swint_ogc.pth": sha256(env["GSAM_ROOT"] + "/gdino_checkpoints/groundingdino_swint_ogc.pth"),
        "sam2_hiera_large.pt": sha256(env["SAM2_CHECKPOINT"]),
        "sam2.1_hiera_large.pt": sha256(env["SAM2_1_CHECKPOINT"]),
        "pipeline.yaml": sha256(env["SAM3D_ROOT"] + "/checkpoints/hf/pipeline.yaml"),
    },
    "sam3d_pipeline_yaml": pathlib.Path(env["SAM3D_ROOT"], "checkpoints/hf/pipeline.yaml").is_file(),
    "hf_token_present": pathlib.Path("/root/.cache/huggingface/token").is_file(),
}
out.mkdir(parents=True, exist_ok=True)
(out / "plan.json").write_text(json.dumps({"pipeline": "setup", "stages": stages}, indent=2) + "\n")
(out / "setup_evidence.json").write_text(json.dumps(document, indent=2) + "\n")
entry = {"item": "box setup", "decision": "accept" if document["all_ok"] else "reject", "use_gpu": True,
         "stages_ok": sum(r["status"] == "ok" for r in results.values()), "stages": len(results)}
if not document["all_ok"]:
    entry["reason"] = ", ".join(f"{k}: {v['status']}" for k, v in results.items() if v["status"] != "ok")
(out / "ledger.json").write_text(json.dumps({"scene_name": "box setup", "entries": [entry]}, indent=2) + "\n")
print(json.dumps({k: v["status"] for k, v in results.items()}))
PY
}

case "$MODE" in
  run) do_run ;;
  report) do_report ;;
  *) echo "usage: box_setup.sh run|report OUT" >&2; exit 2 ;;
esac
