# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Basic CPU image for the sceneagent package: the same pinned packages as the
# CPU CI jobs (.github/workflows/sceneagent-ci.yml), so the test suite, the
# phase-1 contract and the GraspClutter6D scorer run without a GPU.
#
# Not included: the GPU stacks (OmniGibson/Isaac Sim, nerfstudio, COLMAP with
# CUDA, GroundingDINO/SAM, SAM 3D, Depth Anything 3). Those run on the Vast box
# through the gpu-smoke workflow, and scripts/installation/ installs them.
#
#   docker build -t simfoundry-basic .
#   docker run --rm simfoundry-basic                     # CPU test suite
#   docker run --rm -v "$PWD/runs:/opt/simfoundry/runs" simfoundry-basic \
#       python -m sceneagent examples/phase1_place_scene.json --out runs/phase1_place

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg samples video frames; git records the checked-out commit in run evidence;
# libgl1, libglib2.0-0 and libgomp1 are what the
# open3d and OpenCV wheels load at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/simfoundry

# Dependencies first, so a code change does not reinstall them.
COPY requirements-sceneagent-ci.txt requirements-cpu-ci.txt ./
RUN python -m pip install -r requirements-sceneagent-ci.txt -r requirements-cpu-ci.txt

COPY . .

# Non-root by default; runs/ is where the CLI examples write.
RUN useradd --create-home --uid 1000 simfoundry \
    && mkdir -p runs \
    && chown -R simfoundry:simfoundry /opt/simfoundry
USER simfoundry

# The CPU suite, as CI runs it. One test reads `git rev-parse HEAD` from the
# checkout, and the image leaves .git out, so it is deselected here.
CMD ["python", "-m", "pytest", "-q", "tests/sceneagent", "-m", "not gpu and not network", \
     "--deselect", "tests/sceneagent/test_sf5_colmap_evidence.py::test_offline_dry_run_registers_every_native_frame"]
