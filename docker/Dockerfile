# RoboTwin dev image: CUDA + uv + deps from script/requirements.txt
# Build from repo root: docker compose -f docker/docker-compose.headless.yaml build

ARG CUDA_VERSION=12.1
FROM nvidia/cuda:${CUDA_VERSION}.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# System deps: build tools + rendering / sim (SAPIEN / OpenGL / Vulkan)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget tar procps bzip2 ca-certificates git build-essential cmake unzip \
    ffmpeg \
    libgl1 libglib2.0-0 libegl1-mesa libgles2-mesa \
    libxrandr2 libxcursor1 libxdamage1 libxi6 libxtst6 libxext6 \
    libvulkan1 vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Match RoboTwin policy examples; scipy 1.10.x aligns with 3.10
RUN uv venv --python 3.10 /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

WORKDIR /build
COPY script/requirements.txt ./script/requirements.txt

# CUDA wheels from PyTorch index (CPU wheel would break GPU sim / training)
ARG TORCH_CUDA=cu121
RUN uv pip install --extra-index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
    torch==2.4.1 torchvision \
    && uv pip install --extra-index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
    -r script/requirements.txt

WORKDIR /workspace/RoboTwin
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
