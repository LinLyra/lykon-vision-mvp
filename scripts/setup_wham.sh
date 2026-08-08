#!/usr/bin/env bash
set -euo pipefail
mkdir -p external
if [ ! -d external/WHAM ]; then
  git clone --recursive https://github.com/yohanshin/WHAM.git external/WHAM
fi
cat <<'MSG'
WHAM repository cloned to external/WHAM.

Next steps depend on your GPU/CUDA environment and SMPL account access.
Read external/WHAM/README.md and docs for the official install process, then run:

  cd external/WHAM
  bash fetch_demo_data.sh

The official fetch flow requires SMPL/SMPLify registration credentials and downloads model assets under their respective licenses.
MSG
