# RoboTwin dev image: CUDA + uv + deps from pyproject.toml / uv.lock
# Build from repo root: docker compose -f docker/docker-compose.headless.yaml build

ARG CUDA_VERSION=12.1
FROM nvidia/cuda:${CUDA_VERSION}.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# System deps: build tools + rendering / sim (SAPIEN / OpenGL / Vulkan) + git-lfs (CuRobo) + ninja (extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget tar procps bzip2 ca-certificates git git-lfs build-essential cmake unzip \
    ninja-build \
    ffmpeg \
    libgl1 libglib2.0-0 libegl1-mesa libgles2-mesa \
    libxrandr2 libxcursor1 libxdamage1 libxi6 libxtst6 libxext6 \
    libvulkan1 vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

RUN uv venv --python 3.10 /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV CUDA_HOME=/usr/local/cuda
# docker build 默认无 GPU，编译 CUDA 扩展时需显式指定架构，否则 torch cpp_extension 报 arch_list 为空
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"

WORKDIR /build
COPY pyproject.toml uv.lock ./

# Install locked deps (includes nvidia-curobo from source; needs CUDA toolkit matching PyTorch cu121 in lock)
RUN git lfs install \
    && uv sync --frozen

WORKDIR /workspace/RoboTwin
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
