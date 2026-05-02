# =============================================================================
# robotwin Docker image (uv-managed Python environment)
# Build from repo root:  docker build -f docker/Dockerfile .
# =============================================================================

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# -----------------------------------------------------------------------------
# 1. System dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget tar procps bzip2 ca-certificates git git-lfs build-essential cmake unzip tmux \
    python3-dev \
    ninja-build ffmpeg \
    libgl1 libgl1-mesa-glx libglib2.0-0 \
    libegl1-mesa libgles2-mesa libegl1 libegl-mesa0 libosmesa6 \
    libxrandr2 libxcursor1 libxdamage1 libxi6 libxtst6 libxext6 \
    libjpeg-dev libpng-dev xvfb \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Vulkan ICD: SAPIEN/ManiSkill family
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libvulkan1 vulkan-tools libvulkan-dev mesa-vulkan-drivers \
    libegl1 libegl-mesa0 libosmesa6 \
    libjpeg-dev libpng-dev xvfb \
    libxrandr2 libxcursor1 libxdamage1 libxi6 libxtst6 \
 && rm -rf /var/lib/apt/lists/*
COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json

# 
# Uncomment if project needs MuJoCo / robosuite / EGL rendering:
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libx11-6 libxext6 libxrender1 libsm6 libegl1 libosmesa6 \
#     && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 2. uv (fast Python package manager)
# -----------------------------------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
# Alternative (faster, always latest):
# COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# -----------------------------------------------------------------------------
# 3. Python venv via uv
# -----------------------------------------------------------------------------
RUN uv venv --python 3.10 /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

# CUDA C++ extension build environment (auto-set when devel base detected)
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"

# Qt XDG runtime dir (SAPIEN/MPlib load Qt; without this dir Qt warns on every import)
RUN mkdir -p /tmp/runtime-root && chmod 700 /tmp/runtime-root
ENV XDG_RUNTIME_DIR=/tmp/runtime-root
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

# -----------------------------------------------------------------------------
# 4. Non-Python / make-based third-party dependencies → /opt
# -----------------------------------------------------------------------------
# 
# Example:
# WORKDIR /opt/third_party
# RUN git clone https://github.com/<org>/<repo>.git && \
#     cd <repo> && git checkout <commit> && \
#     mkdir build && cd build && cmake .. && make -j$(nproc) && make install
#
# If the dep has Python bindings:
# RUN uv pip install -e /opt/third_party/<repo>

# -----------------------------------------------------------------------------
# 5. Install block — driven by InstallationPlan (<repo>/.nautilus/install_plan.json)
#    or, when no plan exists, by the legacy probe-only Option A/C heuristic.
#
# NOTE: Every `uv sync` / `uv pip install` line carries `--index-strategy
# unsafe-best-match`. uv defaults to "first-index-wins": once any extra index
# (e.g. the PyTorch CUDA wheel index) mentions a package name, uv treats that
# index as the SOLE source and refuses to consult PyPI for a better version.
# This breaks pins like `packaging==23.0` because the PyTorch index ships
# partial mirrors at different versions. `unsafe-best-match` tells uv to
# consider all indexes and pick the best match — what users intuitively
# expect. Harmless when no extra-index is configured. See:
# https://docs.astral.sh/uv/concepts/indexes/
# -----------------------------------------------------------------------------
WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN git lfs install \
    && uv sync --frozen --index-strategy unsafe-best-match

# 
# Slot for downstream sub-skills (benchmark / policy generators) to inject
# extra pip installs via extend_base.py. Leave the placeholder line above —
# it is the insertion anchor.

# -----------------------------------------------------------------------------
# 6. Extension point for sub-skills (benchmark / policy / robot env-generator)
# -----------------------------------------------------------------------------
# Sub-skills insert RUN lines ABOVE this marker via their extend_base.py script.
# Do NOT remove the marker line — it is how downstream tooling locates the insertion point.
# <<<EXTENSION_INSERT_ABOVE>>>

# -----------------------------------------------------------------------------
# 7. Workspace (project source mounted at runtime, not copied)
# -----------------------------------------------------------------------------
WORKDIR /workspace/RoboTwin
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
