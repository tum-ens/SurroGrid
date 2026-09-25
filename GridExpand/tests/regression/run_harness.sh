#!/usr/bin/env bash
# End-to-end regression run of the synthetic pipeline against a SANDBOX database.
#
# Usage: tests/regression/run_harness.sh <out_dir> [pre] [heur] [opt]
#   Run from GridExpand/. The .env of this checkout (or $GRIDEXPAND_ENV_FILE) must point to the sandbox:
#   DB_HOST=$HARNESS_DB_HOST (127.0.0.1), DB_PORT=$HARNESS_DB_PORT (55439), DB_NAME starting with
#   $HARNESS_DB_PREFIX (sg_). The database is DROPPED and recreated from $HARNESS_TEMPLATE_DB.
#   HARNESS_QUICK=1 runs one grid only; HARNESS_SCENARIO overrides the scenario YAML.
#   Then: uv run python tests/regression/compare.py <reference>/snapshot.pkl <out_dir>/snapshot.pkl
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
PROJECT=$(realpath "$HERE/../..")
OUT=$(realpath -m "$1"); shift
CASES=${*:-pre heur opt}
CONTAINER=${HARNESS_CONTAINER:-pylovo-fable-sandbox}
TEMPLATE=${HARNESS_TEMPLATE_DB:-surrogrid_fable_base}
ENV_FILE=${GRIDEXPAND_ENV_FILE:-$PROJECT/.env}
value() { grep -E "^$1\s*=" "$ENV_FILE" | head -1 | cut -d= -f2- | sed -E 's/^\s*"?//; s/"?\s*(#.*)?$//'; }
[ "$(value DB_HOST)" = "${HARNESS_DB_HOST:-127.0.0.1}" ] || { echo "refusing: $ENV_FILE is not the sandbox host"; exit 3; }
[ "$(value DB_PORT)" = "${HARNESS_DB_PORT:-55439}" ] || { echo "refusing: $ENV_FILE is not the sandbox port"; exit 3; }
DB=$(value DB_NAME)
case "$DB" in "${HARNESS_DB_PREFIX:-sg_}"*) ;; *) echo "refusing: unexpected DB name $DB"; exit 3;; esac

PSQL="docker exec $CONTAINER psql -U sandbox -d sandbox -v ON_ERROR_STOP=1 -q"
$PSQL -c "select pg_terminate_backend(pid) from pg_stat_activity where datname='$DB' and pid<>pg_backend_pid();" >/dev/null
$PSQL -c "DROP DATABASE IF EXISTS \"$DB\";" -c "CREATE DATABASE \"$DB\" TEMPLATE $TEMPLATE;"
rm -rf "$OUT"; mkdir -p "$OUT/work"
unset VIRTUAL_ENV
export PYTHONHASHSEED=0 GRIDEXPAND_WORK_DIR="$OUT/work" GRIDEXPAND_ENV_FILE="$ENV_FILE"
cd "$PROJECT"
COMMON=(--ags 9184137 --pylovo-version-id 1 --min-buildings 60
  --workers 2 --step2-cpus 2 --step3-cpus 4 --step3-max-cpus 8 --step4-cpus 2
  --scenario-config "${HARNESS_SCENARIO:-$HERE/scenario_sandbox.yaml}" --timeframe-mode max_base_electricity_demand_week
  --powerflow-output both --case-qualified-output --no-pilot-gate --profile-seed 481527)
if [ "${HARNESS_QUICK:-0}" = "1" ]; then COMMON+=(--start-index 3 --limit 1); fi
for c in $CASES; do
  case $c in
    pre)  ARGS=(--model-case pre --profiles status_quo) ;;
    heur) ARGS=(--model-case post-hems-heuristic --profiles all) ;;
    opt)  ARGS=(--model-case post-hems-optimized --profiles all) ;;
    *) echo "unknown case $c"; exit 2 ;;
  esac
  start=$(date +%s)
  uv run --frozen gridexpand synthetic "${COMMON[@]}" "${ARGS[@]}" --run-dir "$OUT/$c" > "$OUT/$c.log" 2>&1 \
    || echo "CASE FAILED: $c (see $OUT/$c.log)"
  echo "$c: $(( $(date +%s) - start )) s"
done
uv run --frozen python "$HERE/snapshot.py" "$DB" "$OUT/snapshot.pkl"
