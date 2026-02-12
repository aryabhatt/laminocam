#!/bin/bash
# Setup script for lamino - adds bin to PATH and lib to LD_LIBRARY_PATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PATH="${SCRIPT_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${SCRIPT_DIR}/lib:${LD_LIBRARY_PATH}"

echo "Lamino environment setup complete"
echo "  PATH: ${SCRIPT_DIR}/bin"
echo "  LD_LIBRARY_PATH: ${SCRIPT_DIR}/lib"
