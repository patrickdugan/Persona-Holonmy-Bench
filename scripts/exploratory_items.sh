#!/usr/bin/env bash
# Exploratory (not preregistered): item vs seed-range check. Runs the three
# calibration probe families on the confirmatory seeds 20000-20031 with the
# same 4B settings, after scripts/confirmatory_run.sh has finished.
set -u
cd "$(dirname "$0")/.." || exit 1
V=/c/Users/patri/AppData/Local/Programs/vramorama/vramorama.exe
OUT=runs/confirm
while [ "$#" -gt 0 ] && kill -0 "$1" 2>/dev/null; do sleep 60; done
while ! grep -q "confirmatory pipeline finished" "$OUT"/pipeline-*.out 2>/dev/null; do sleep 120; done
for g in gothic pirate noir; do
  name=x-4b-origprobes-$g
  [ -f "$OUT/$name.done" ] && continue
  [ -e "$OUT/$name" ] && mv "$OUT/$name" "$OUT/$name.partial.$(date +%s)"
  "$V" run --vram 11G --owner "phb-$name" -- .venv/Scripts/python.exe -u scripts/h1_calibration.py \
    --model 'C:\Users\patri\.cache\huggingface\hub\models--Qwen--Qwen3-4B-Instruct-2507\snapshots\cdbee75f17c01a7cc42f958dc650907174af0554' \
    --revision cdbee75f17c01a7cc42f958dc650907174af0554 --dtype bfloat16 --seed 20000 --repetitions 32 \
    --genres $g --system-variants default --conditions ordered,neutral,lexical_match,lexical_neutral \
    --phases exit --generate-phases "" --probes flaky_ci,parser_validator,stale_cache \
    --output-dir "$OUT/$name" > "$OUT/$name.log" 2>&1 && touch "$OUT/$name.done"
done
echo "exploratory items finished"
