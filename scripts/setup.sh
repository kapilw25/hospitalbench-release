#!/usr/bin/env bash
# setup.sh: one command that makes the Python environment and fetches the room files.
# Needs: Linux x86_64, Python 3.12, an NVIDIA RTX GPU with a current driver, internet (pip, NVIDIA content server).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3.12}
$PY -c 'import sys; assert sys.version_info[:2] == (3, 12), "Isaac Sim 6.1 pip needs Python 3.12"'
[ -d .venv ] || $PY -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install "isaacsim[all,extscache]==6.1.0.0" --extra-index-url https://pypi.nvidia.com
pip install -r requirements.txt
python scripts/fetch_rooms.py "$@"
python scripts/fetch_assets.py --list | tail -1 || true   # size of the SimReady assets; the first run of run_scripted.py / run_model.py fetches the ones its chore uses (or: python scripts/fetch_assets.py)
echo
echo "Setup done. Before the first run read the NVIDIA Omniverse EULA and, if you accept it:  export OMNI_KIT_ACCEPT_EULA=YES"
echo "Then:  .venv/bin/python run_scripted.py --task T1 --state 0"
