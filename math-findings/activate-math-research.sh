#!/usr/bin/env bash

# Source this file from any shell to activate the isolated Sentinel math stack.
sentinel_math_project="/Users/anthonycavero/Documents/Startup/project-sentinel"

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source this script instead of executing it:"
  echo "  source \"${BASH_SOURCE[0]}\""
  exit 2
fi

source "${sentinel_math_project}/.venv-math-research/bin/activate"
export ELAN_HOME="${sentinel_math_project}/.math-research-tools/elan"
export PATH="${ELAN_HOME}/bin:${PATH}"
export MLFLOW_TRACKING_URI="sqlite:///${sentinel_math_project}/.math-research-tools/mlflow.db"

echo "Sentinel math research environment active."
echo "Python: $(python --version 2>&1)"
echo "Lean: $(lean --version | head -n 1)"
