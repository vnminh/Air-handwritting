#!/usr/bin/env bash
# Pull benchmark results (and checkpoints of completed runs) from the training
# server and push results to GitHub whenever
# the number of completed runs grows. Only benchmarks/ is staged or committed.
#
#   scripts/sync_results.sh [interval_seconds]
set -u
REMOTE="${REMOTE:-root@88.90.102.66}"
PORT="${PORT:-41154}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/Air-handwritting/benchmarks/}"
INTERVAL="${1:-300}"
cd "$(dirname "$0")/.."
last=-1

while true; do
  rsync -az --exclude 'best.pt' --exclude '.cache/' -e "ssh -p $PORT -o ConnectTimeout=20" \
    "$REMOTE:$REMOTE_DIR" benchmarks/ 2>/dev/null
  # Download the final checkpoint of every completed run (git ignores it).
  for result in $(grep -l '"status": "complete"' benchmarks/*/*/seed_*/result.json 2>/dev/null); do
    dir=${result%/result.json}
    if [ ! -f "$dir/best.pt" ] && rsync -az -e "ssh -p $PORT -o ConnectTimeout=20" \
        "$REMOTE:${REMOTE_DIR%/}/${dir#benchmarks/}/best.pt" "$dir/" 2>/dev/null; then
      echo "$(date '+%F %T') downloaded $dir/best.pt"
    fi
  done
  completed=$(grep -l '"status": "complete"' benchmarks/*/*/seed_*/result.json 2>/dev/null | wc -l)
  completed=$((completed + $(ls benchmarks/robustness/*.json 2>/dev/null | wc -l)))
  if [ "$completed" != "$last" ]; then
    git add benchmarks
    git diff --cached --quiet -- benchmarks \
      || git commit -q -m "results: sync benchmarks ($completed completed runs)" -- benchmarks
    last=$completed
  fi
  # Retry pushing until the remote has every local commit.
  if [ -n "$(git log origin/main..HEAD --oneline 2>/dev/null)" ]; then
    git pull -q --rebase --autostash origin main \
      && git push -q origin main \
      && echo "$(date '+%F %T') pushed ($completed completed runs)"
  fi
  sleep "$INTERVAL"
done
