#!/bin/bash

set -euo pipefail

THIS_DIR="$(dirname "$(realpath "$0")")"
ROOT_DIR="$(dirname "$THIS_DIR")"

for i in {1..14}; do
  "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/script.py"
done
