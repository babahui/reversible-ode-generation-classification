#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${GITHUB_REPO:-babahui/reversible-ode-generation-classification}"
BRANCH="${GITHUB_BRANCH:-main}"
cd "$ROOT"

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || git init
git add .gitignore README.md WEIGHTS.md requirements.txt requirements-eval.txt \
  environment.yml configs results scripts unified_transport tests \
  -- '*.py' '*.md' '*.sh' '*.yml' '*.json' '*.txt'
if ! git diff --cached --quiet; then
  git commit -m "Prepare reproducible reversible ODE release"
fi

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "Local release commit is ready: $(git rev-parse --short HEAD)"
  echo "No GITHUB_TOKEN was provided, so no remote repository was created or pushed."
  echo "Set GITHUB_TOKEN with repo creation permission and rerun this script."
  echo "Target repository: https://github.com/$REPO"
  exit 0
fi

owner="${REPO%%/*}"
name="${REPO#*/}"
if [[ "$owner" == "$name" || -z "$owner" || -z "$name" ]]; then
  echo "GITHUB_REPO must be in owner/name form" >&2
  exit 2
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  payload=$(printf '{"name":"%s","description":"Reversible ODE for joint image generation and classification with low-frequency Gaussian noise, OT and CE","private":false}' "$name")
  curl --fail-with-body --silent --show-error \
    -X POST "https://api.github.com/user/repos" \
    -H "Authorization: Bearer ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    -d "$payload" >/dev/null || {
      echo "GitHub repository creation failed. Check token scope and whether $REPO already exists." >&2
      exit 3
    }
  git remote add origin "https://github.com/$REPO.git"
fi

git -c http.extraHeader="Authorization: Bearer ${GITHUB_TOKEN}" \
  push --set-upstream origin "HEAD:$BRANCH"
echo "Published https://github.com/$REPO"
