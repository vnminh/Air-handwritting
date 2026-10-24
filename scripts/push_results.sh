#!/usr/bin/env bash
# Run on the training server: commit benchmarks/ and push to GitHub whenever
# the number of completed runs or robustness sweeps grows. Checkpoints stay
# on the server (.gitignore). Requires a GitHub deploy key with write access.
#
#   scripts/push_results.sh [interval_seconds]
set -u
INTERVAL="${1:-300}"
cd "$(dirname "$0")/.."
last=-1

while true; do
  completed=$(grep -l '"status": "complete"' benchmarks/*/*/seed_*/result.json 2>/dev/null | wc -l)
  completed=$((completed + $(ls benchmarks/robustness/*.json 2>/dev/null | wc -l)))
  if [ "$completed" != "$last" ]; then
    git add benchmarks
    git diff --cached --quiet -- benchmarks \
      || git commit -q -m "results: sync benchmarks ($completed completed runs)" -- benchmarks
    last=$completed
  fi
  if [ -n "$(git log origin/main..HEAD --oneline 2>/dev/null)" ]; then
    git pull -q --rebase --autostash origin main \
      && git push -q origin main \
      && echo "$(date '+%F %T') pushed ($completed completed runs)"
  fi
  sleep "$INTERVAL"
done
