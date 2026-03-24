#!/bin/bash
set -e

export PATH="/opt/venv/bin:/usr/local/bin:${PATH:-/usr/bin:/bin}"
export VIRTUAL_ENV="/opt/venv"

# Editable install only if root package metadata exists
if [ -f "/workspace/RoboTwin/pyproject.toml" ]; then
    echo ">> Installing editable package (pyproject)..."
    cd /workspace/RoboTwin && uv pip install -e . && cd - > /dev/null
elif [ -f "/workspace/RoboTwin/setup.py" ]; then
    echo ">> Installing editable package (setup.py)..."
    cd /workspace/RoboTwin && uv pip install -e . --no-deps && cd - > /dev/null
fi

if [ "${INSTALL_CLAUDE_CODE}" = "1" ]; then
    echo ">> Installing Claude Code CLI..."
    if curl -fsSL https://claude.ai/install.sh | bash 2>/dev/null; then
        export PATH="${HOME}/.local/bin:${PATH}"
        echo ">> Claude Code CLI installed. Run 'claude' to start."
    else
        echo ">> Claude Code CLI install skipped (network unavailable)."
    fi
fi

echo ">> Ready. Repo: /workspace/RoboTwin (see README / official doc for assets & tasks)."
exec "$@"
