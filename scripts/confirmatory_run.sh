#!/usr/bin/env bash
# Confirmatory run for docs/preregistration.md. Stages are idempotent: a stage
# with a runs/confirm/<stage>.done marker is skipped; an incomplete stage
# directory is moved aside (kept, not deleted) and rerun.
# GPU stages go through vramorama leases. Run from the repo root.
set -u
cd "$(dirname "$0")/.." || exit 1
V=/c/Users/patri/AppData/Local/Programs/vramorama/vramorama.exe
PY=.venv/Scripts/python.exe
PHB=.venv/Scripts/phb.exe
M4='C:\Users\patri\.cache\huggingface\hub\models--Qwen--Qwen3-4B-Instruct-2507\snapshots\cdbee75f17c01a7cc42f958dc650907174af0554'
R4=cdbee75f17c01a7cc42f958dc650907174af0554
M17='C:\Users\patri\.cache\huggingface\hub\models--Qwen--Qwen3-1.7B\snapshots\70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
R17=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
OUT=runs/confirm
CONF=retry_backoff,schema_migration,memory_leak,rate_limit,feature_flag,log_rotation
ALLP=flaky_ci,parser_validator,stale_cache,$CONF
COND=ordered,neutral,lexical_match,lexical_neutral
CONTR=lexical_match-lexical_neutral,lexical_match-neutral,lexical_neutral-neutral
mkdir -p "$OUT"

stage() {  # stage <name> <vram> <command...>
  local name=$1 vram=$2; shift 2
  if [ -f "$OUT/$name.done" ]; then echo "skip $name"; return 0; fi
  if [ -e "$OUT/$name" ]; then mv "$OUT/$name" "$OUT/$name.partial.$(date +%s)"; fi
  "$V" run --vram "$vram" --owner "phb-confirm-$name" -- "$@" > "$OUT/$name.log" 2>&1
  local status=$?
  echo "$name exit=$status"
  [ $status -eq 0 ] && touch "$OUT/$name.done"
  return 0
}

decomp() {  # decomp <name> <model> <rev> <dtype> <genre> <variants> <phases> <seed> <reps> [extra...]
  local name=$1 model=$2 rev=$3 dtype=$4 genre=$5 variants=$6 phases=$7 seed=$8 reps=$9
  shift 9
  stage "$name" 11G $PY -u scripts/h1_calibration.py --model "$model" --revision "$rev" \
    --dtype "$dtype" --seed "$seed" --repetitions "$reps" --genres "$genre" \
    --system-variants "$variants" --conditions $COND --contrasts $CONTR \
    --phases "$phases" --generate-phases "" --probes $CONF --output-dir "$OUT/$name" "$@"
}

# Gate 1: baseline placebo (seeds 20000-20003).
decomp 4b-placebo "$M4" $R4 bfloat16 gothic,pirate,noir default baseline 20000 4

# H1-H3 (primary, 4B): peak and exit, both system prompts, per genre.
for g in gothic pirate noir; do
  decomp 4b-main-$g "$M4" $R4 bfloat16 $g default,plain peak,exit 20000 32
done

# H4 (Q6 mechanism): own case file, seeds 20000-20063, all nine families.
if [ ! -f "$OUT/q6-cases.jsonl" ]; then
  $PHB generate --variation seeded_v1 --seed 20000 --repetitions 64 --genres gothic \
    --system-variants default --conditions ordered,lexical_match --phases exit \
    --probes $ALLP --output "$OUT/q6-cases.jsonl"
  $PHB sweep-plan --input "$OUT/q6-cases.jsonl" --layers 18,9,27 --alphas 1 \
    --sources ordered --reference lexical_match --heldout-probes $CONF \
    --holdout-fraction 0.5 --split-seed 20261025 --readout copy_free --n-nulls 8 \
    --phases exit --output "$OUT/q6-plan.json"
fi
stage q6-sweep 11G $PHB sweep --model "$M4" --input "$OUT/q6-cases.jsonl" \
  --layers 18,9,27 --alphas 1 --sources ordered --reference lexical_match \
  --heldout-probes $CONF --holdout-fraction 0.5 --split-seed 20261025 \
  --readout copy_free --n-nulls 8 --phases exit --device cuda --dtype bfloat16 \
  --max-total-tokens 2048 --output "$OUT/q6-sweep.jsonl"
[ -f "$OUT/q6-sweep.done" ] && $PHB sweep-report --input "$OUT/q6-sweep.jsonl" \
  --output-prefix "$OUT/q6-report"

# Replication (1.7B, float32, --no-thinking): H1-H3 cells.
for g in gothic pirate noir; do
  decomp 17b-main-$g "$M17" $R17 float32 $g default,plain peak,exit 20000 32 --no-thinking
done
# Descriptive time course (4B, default system): washout_1 and washout_2.
for g in gothic pirate noir; do
  decomp 4b-washout-$g "$M4" $R4 bfloat16 $g default washout_1,washout_2 20000 32
done

# Q5 sampled behaviour (4B, exit, default system, seeds 20000-20015).
if [ ! -f "$OUT/q5-cases.jsonl" ]; then
  $PHB generate --variation seeded_v1 --seed 20000 --repetitions 16 \
    --genres gothic,pirate,noir --system-variants default --conditions $COND \
    --phases exit --probes $CONF --output "$OUT/q5-cases.jsonl"
fi
stage q5-sample 11G $PHB sample-rates --model "$M4" --input "$OUT/q5-cases.jsonl" \
  --n-samples 20 --temperature 1.0 --max-new-tokens 64 --device cuda --dtype bfloat16 \
  --max-total-tokens 2048 --output "$OUT/q5-sample.jsonl"

echo "confirmatory pipeline finished"
