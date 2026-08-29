#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${PROJECT_ROOT}/runs/audit-pycache}"

if rg -n '/DATA/|/home/' \
  "${PROJECT_ROOT}" \
  -g '!scripts/audit_release.sh' \
  -g '!runs/**'; then
  echo "Cluster-specific absolute path detected." >&2
  exit 1
fi

if rg -n '[\p{Han}]' "${PROJECT_ROOT}" -g '!runs/**'; then
  echo "Non-English annotation detected." >&2
  exit 1
fi

if rg -n '[\x{3000}-\x{303F}\x{FF00}-\x{FFEF}]' "${PROJECT_ROOT}" -g '!runs/**'; then
  echo "Full-width annotation punctuation detected." >&2
  exit 1
fi

if rg -n 'wandb\.ai/|logger:.*wandb|logger=.*wandb|trainer\.logger.*wandb' \
  "${PROJECT_ROOT}" \
  -g '!scripts/audit_release.sh' \
  -g '!runs/**'; then
  echo "External tracking account link or default tracking logger detected." >&2
  exit 1
fi

if rg -n -i \
  '(api[_-]?key|access[_-]?key|secret|password|authorization)[[:space:]]*[:=][[:space:]]*["'\''][^"'\'']+["'\'']' \
  "${PROJECT_ROOT}" \
  -g '!scripts/audit_release.sh' \
  -g '!runs/**'; then
  echo "Literal credential-like assignment detected." >&2
  exit 1
fi

if git -C "${PROJECT_ROOT}" ls-files --cached --others --exclude-standard \
  | rg -q '(^|/)([^/]+\.(key|pem|lic)|\.env)$'; then
  echo "Credential-like file detected." >&2
  exit 1
fi

"${PYTHON_BIN}" -m compileall -q \
  "${PROJECT_ROOT}/verl" \
  "${PROJECT_ROOT}/scripts"

echo "[PASS] release audit"
