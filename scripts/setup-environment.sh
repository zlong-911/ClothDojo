#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${project_root}/.venv"
python_bin="${PYTHON:-python3.11}"

if [[ ! -x "${venv_dir}/bin/python" ]]; then
    "${python_bin}" -m venv "${venv_dir}"
fi

python_version="$("${venv_dir}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${python_version}" != "3.11" ]]; then
    echo "ClothDojo requires Python 3.11; ${venv_dir} uses ${python_version}." >&2
    exit 1
fi

"${venv_dir}/bin/python" -m pip install -r "${project_root}/requirements.txt"

# Isaac Sim 5 pins shared packages such as torch, fsspec, scipy and numpy to
# versions that conflict with ClothDojo's audited training stack. Install only
# the binary runtime distributions and extension caches; requirements.txt owns
# all shared Python dependencies in this single environment.
"${venv_dir}/bin/python" -m pip install \
    --extra-index-url https://pypi.nvidia.com \
    --no-deps \
    'isaacsim==5.0.0' \
    'isaacsim-app==5.0.0' \
    'isaacsim-core==5.0.0' \
    'isaacsim-replicator==5.0.0' \
    'isaacsim-extscache-kit==5.0.0' \
    'isaacsim-extscache-kit-sdk==5.0.0' \
    'isaacsim-extscache-physics==5.0.0' \
    'isaacsim-kernel==5.0.0' \
    'omniverse-kit==107.3.1.206797'

"${venv_dir}/bin/python" -m pip install --no-deps -e "${project_root}"
