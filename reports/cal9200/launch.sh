#!/usr/bin/env bash
# 9200+ calibration on Qwen3-4B-Instruct: one run per genre, sequential, under vramorama.
cd /c/Users/patri/Documents/GitHub/MoralityLabAI/Persona-Holonmy-Bench || exit 1
V=/c/Users/patri/AppData/Local/Programs/vramorama/vramorama.exe
M='C:\Users\patri\.cache\huggingface\hub\models--Qwen--Qwen3-4B-Instruct-2507\snapshots\cdbee75f17c01a7cc42f958dc650907174af0554'
for genre in gothic pirate noir; do
  out=runs/cal9200/$genre
  if [ -f "$out/summary.json" ]; then echo "$genre already done"; continue; fi
  rm -rf "$out"
  "$V" run --vram 11G --owner "phb-cal9200-$genre" -- .venv/Scripts/python.exe -u scripts/h1_calibration.py \
    --model "$M" --revision cdbee75f17c01a7cc42f958dc650907174af0554 --dtype bfloat16 \
    --seed 9200 --repetitions 16 --genres "$genre" --system-variants default,plain \
    --conditions ordered,neutral,lexical_match,lexical_neutral \
    --contrasts lexical_match-lexical_neutral,lexical_match-neutral,lexical_neutral-neutral \
    --phases peak,exit,washout_2 --generate-phases exit --max-new-tokens 64 \
    --probes flaky_ci,parser_validator,stale_cache --output-dir "$out" \
    > "runs/cal9200/$genre.log" 2>&1
  echo "$genre exit=$?"
done
